export interface DocumentChunk {
  text: string;
  page: number;
  source: string;
  index: number;
}

export interface ParsedDocument {
  name: string;
  type: string;
  pageCount: number;
  chunks: DocumentChunk[];
}

const CHUNK_SIZE = 300;
const CHUNK_OVERLAP = 50;

function splitIntoChunks(
  text: string,
  page: number,
  source: string,
  startIndex: number
): DocumentChunk[] {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks: DocumentChunk[] = [];
  const step = Math.max(1, CHUNK_SIZE - CHUNK_OVERLAP);

  for (let i = 0; i < words.length; i += step) {
    const slice = words.slice(i, i + CHUNK_SIZE);
    if (slice.length < 15 && i > 0) continue;
    chunks.push({
      text: slice.join(" "),
      page,
      source,
      index: startIndex + chunks.length,
    });
  }
  return chunks;
}

export async function parsePDF(buffer: Buffer, filename: string): Promise<ParsedDocument> {
  const pdfParse = (await import("pdf-parse")).default;
  const data = await pdfParse(buffer);
  const pages = data.text.split(/\f/);
  const allChunks: DocumentChunk[] = [];

  for (let i = 0; i < pages.length; i++) {
    const cleaned = pages[i].replace(/\s+/g, " ").trim();
    if (!cleaned) continue;
    const chunks = splitIntoChunks(cleaned, i + 1, filename, allChunks.length);
    allChunks.push(...chunks);
  }

  return {
    name: filename,
    type: "pdf",
    pageCount: data.numpages,
    chunks: allChunks,
  };
}

export async function parseDOCX(buffer: Buffer, filename: string): Promise<ParsedDocument> {
  const mammoth = await import("mammoth");
  const result = await mammoth.extractRawText({ buffer });
  const paragraphs = result.value.split(/\n\n+/).filter(Boolean);
  const allChunks: DocumentChunk[] = [];

  for (let i = 0; i < paragraphs.length; i++) {
    const cleaned = paragraphs[i].replace(/\s+/g, " ").trim();
    if (!cleaned) continue;
    const chunks = splitIntoChunks(cleaned, i + 1, filename, allChunks.length);
    allChunks.push(...chunks);
  }

  return {
    name: filename,
    type: "docx",
    pageCount: paragraphs.length,
    chunks: allChunks,
  };
}

export async function parseXLSX(buffer: Buffer, filename: string): Promise<ParsedDocument> {
  const XLSX = await import("xlsx");
  const workbook = XLSX.read(buffer, { type: "buffer" });
  const allChunks: DocumentChunk[] = [];

  for (let si = 0; si < workbook.SheetNames.length; si++) {
    const sheetName = workbook.SheetNames[si];
    const sheet = workbook.Sheets[sheetName];
    const csv = XLSX.utils.sheet_to_csv(sheet);
    const rows = csv.split("\n").filter((r: string) => r.trim());
    const text = `Sheet: ${sheetName}\n${rows.join("\n")}`;
    const chunks = splitIntoChunks(text, si + 1, filename, allChunks.length);
    allChunks.push(...chunks);
  }

  return {
    name: filename,
    type: "xlsx",
    pageCount: workbook.SheetNames.length,
    chunks: allChunks,
  };
}

export async function parsePPTX(buffer: Buffer, filename: string): Promise<ParsedDocument> {
  const JSZip = (await import("xlsx")).default;
  // PPTX is a ZIP containing XML slides
  // We'll use a lightweight approach: extract text from slide XML
  const zip = JSZip.read(buffer, { type: "buffer" });
  const allChunks: DocumentChunk[] = [];
  let slideNum = 0;

  // PPTX slides are in xl/ but we need to handle pptx differently
  // For simplicity, treat the entire content as text
  const sheets = Object.keys(zip.Sheets || {});
  if (sheets.length > 0) {
    // Fallback: treat as spreadsheet-like
    for (const name of sheets) {
      slideNum++;
      const csv = JSZip.utils.sheet_to_csv(zip.Sheets[name]);
      const chunks = splitIntoChunks(csv, slideNum, filename, allChunks.length);
      allChunks.push(...chunks);
    }
  }

  return {
    name: filename,
    type: "pptx",
    pageCount: slideNum,
    chunks: allChunks.length > 0 ? allChunks : [{
      text: "Could not extract text from this presentation file. Try converting to PDF first.",
      page: 1,
      source: filename,
      index: 0,
    }],
  };
}

export async function parseDocument(
  buffer: Buffer,
  filename: string
): Promise<ParsedDocument> {
  const ext = filename.toLowerCase().split(".").pop() || "";

  switch (ext) {
    case "pdf":
      return parsePDF(buffer, filename);
    case "docx":
    case "doc":
      return parseDOCX(buffer, filename);
    case "xlsx":
    case "xls":
    case "csv":
      return parseXLSX(buffer, filename);
    case "pptx":
    case "ppt":
      return parsePPTX(buffer, filename);
    case "txt":
    case "md": {
      const text = buffer.toString("utf-8");
      const chunks = splitIntoChunks(text, 1, filename, 0);
      return { name: filename, type: ext, pageCount: 1, chunks };
    }
    default:
      throw new Error(`Unsupported file type: .${ext}`);
  }
}

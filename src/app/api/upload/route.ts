import { NextRequest, NextResponse } from "next/server";
import { parseDocument } from "@/lib/document-processor";

export async function POST(req: NextRequest) {
  try {
    const formData = await req.formData();
    const file = formData.get("file") as File | null;

    if (!file) {
      return NextResponse.json({ error: "No file provided" }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    const doc = await parseDocument(buffer, file.name);

    return NextResponse.json({
      name: doc.name,
      type: doc.type,
      pageCount: doc.pageCount,
      chunkCount: doc.chunks.length,
      chunks: doc.chunks,
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Failed to parse document";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}

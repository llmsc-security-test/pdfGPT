import type { DocumentChunk } from "./document-processor";

export interface SearchResult {
  chunk: DocumentChunk;
  score: number;
}

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2);
}

function computeTFIDF(
  chunks: DocumentChunk[]
): { vectors: Map<string, number>[]; idf: Map<string, number> } {
  const docCount = chunks.length;
  const docFreq = new Map<string, number>();
  const tokenized = chunks.map((c) => tokenize(c.text));

  for (const tokens of tokenized) {
    const unique = new Set(tokens);
    for (const token of unique) {
      docFreq.set(token, (docFreq.get(token) || 0) + 1);
    }
  }

  const idf = new Map<string, number>();
  for (const [term, freq] of docFreq) {
    idf.set(term, Math.log((docCount + 1) / (freq + 1)) + 1);
  }

  const vectors = tokenized.map((tokens) => {
    const tf = new Map<string, number>();
    for (const t of tokens) {
      tf.set(t, (tf.get(t) || 0) + 1);
    }
    const vec = new Map<string, number>();
    for (const [term, count] of tf) {
      const tfidf = (count / tokens.length) * (idf.get(term) || 0);
      vec.set(term, tfidf);
    }
    return vec;
  });

  return { vectors, idf };
}

function cosineSim(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (const [key, val] of a) {
    dot += val * (b.get(key) || 0);
    normA += val * val;
  }
  for (const val of b.values()) {
    normB += val * val;
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

export function searchChunks(
  query: string,
  chunks: DocumentChunk[],
  topK: number = 5
): SearchResult[] {
  if (chunks.length === 0) return [];

  const allDocs = [...chunks, { text: query, page: 0, source: "", index: -1 }];
  const { vectors } = computeTFIDF(allDocs);
  const queryVec = vectors[vectors.length - 1];
  const docVectors = vectors.slice(0, -1);

  const scored: SearchResult[] = docVectors.map((vec, i) => ({
    chunk: chunks[i],
    score: cosineSim(queryVec, vec),
  }));

  scored.sort((a, b) => b.score - a.score);
  const results = scored.slice(0, topK);
  if (results.every((r) => r.score === 0) && chunks.length > 0) {
    return chunks.slice(0, topK).map((c, i) => ({ chunk: c, score: 0.1 - i * 0.01 }));
  }
  return results;
}

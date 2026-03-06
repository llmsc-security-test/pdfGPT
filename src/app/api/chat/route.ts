import { NextRequest } from "next/server";
import { streamText } from "ai";
import { createOpenAI } from "@ai-sdk/openai";
import { createAnthropic } from "@ai-sdk/anthropic";
import { createGoogleGenerativeAI } from "@ai-sdk/google";
import { searchChunks } from "@/lib/search";
import type { DocumentChunk } from "@/lib/document-processor";

function getProvider(modelId: string, apiKey?: string) {
  const [providerName, modelName] = modelId.split(":");

  switch (providerName) {
    case "google": {
      const google = createGoogleGenerativeAI({
        apiKey: apiKey || process.env.GOOGLE_GENERATIVE_AI_API_KEY || "",
      });
      return google(modelName);
    }
    case "groq": {
      const groq = createOpenAI({
        baseURL: "https://api.groq.com/openai/v1",
        apiKey: apiKey || process.env.GROQ_API_KEY || "",
      });
      return groq(modelName);
    }
    case "openai": {
      const openai = createOpenAI({
        apiKey: apiKey || process.env.OPENAI_API_KEY || "",
      });
      return openai(modelName);
    }
    case "anthropic": {
      const anthropic = createAnthropic({
        apiKey: apiKey || process.env.ANTHROPIC_API_KEY || "",
      });
      return anthropic(modelName);
    }
    default:
      throw new Error(`Unknown provider: ${providerName}`);
  }
}

function buildRAGPrompt(question: string, context: string): string {
  return `You are a document analysis assistant. Answer the question using ONLY the provided document excerpts. Cite sources using [Page N] format.

Document excerpts:
${context}

Question: ${question}

Rules:
- Only use information from the excerpts above.
- Cite page numbers for every claim using [Page N].
- If the information is not in the excerpts, say so clearly.
- Be thorough and accurate.`;
}

function buildAgenticPrompt(question: string, context: string): string {
  return `You are an expert document analyst performing deep analysis. You have been given excerpts from a document. Perform a thorough, multi-step analysis.

Document excerpts:
${context}

Question: ${question}

Perform the following steps:
1. ANALYZE: Identify the key aspects of the question that need to be addressed.
2. EXTRACT: Pull all relevant information from the document excerpts.
3. SYNTHESIZE: Combine findings into a coherent analysis.
4. EVALUATE: Assess the completeness and confidence of your answer.
5. RESPOND: Provide a comprehensive answer with [Page N] citations.

Format your response with clear sections. Be thorough and analytical. If information is missing, identify what gaps exist.`;
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {
      question,
      chunks,
      modelId,
      apiKey,
      mode = "rag",
    } = body as {
      question: string;
      chunks: DocumentChunk[];
      modelId: string;
      apiKey?: string;
      mode?: "rag" | "agentic";
    };

    if (!question) {
      return new Response(JSON.stringify({ error: "No question" }), { status: 400 });
    }

    const results = searchChunks(question, chunks || [], 8);

    const context = results
      .map((r) => `[Page ${r.chunk.page}] ${r.chunk.text}`)
      .join("\n\n");

    if (!context) {
      return new Response(
        JSON.stringify({ error: "No relevant content found. Try a different question." }),
        { status: 200 }
      );
    }

    const prompt =
      mode === "agentic"
        ? buildAgenticPrompt(question, context)
        : buildRAGPrompt(question, context);

    const model = getProvider(modelId, apiKey);

    const result = streamText({
      model,
      messages: [{ role: "user", content: prompt }],
      temperature: 0.3,
      maxTokens: 2048,
    });

    return result.toDataStreamResponse();
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Chat error";
    console.error("Chat API error:", message);
    return new Response(
      JSON.stringify({ error: message }),
      { status: 500, headers: { "Content-Type": "application/json" } }
    );
  }
}

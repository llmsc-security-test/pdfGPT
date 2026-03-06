export interface ModelOption {
  id: string;
  name: string;
  provider: string;
  requiresKey: boolean;
  keyEnvVar: string;
  free: boolean;
  description: string;
}

export const MODEL_LIST: ModelOption[] = [
  {
    id: "google:gemini-2.0-flash",
    name: "Gemini 2.0 Flash",
    provider: "Google",
    requiresKey: true,
    keyEnvVar: "GOOGLE_GENERATIVE_AI_API_KEY",
    free: true,
    description: "Free - fast and capable (get key at aistudio.google.com)",
  },
  {
    id: "google:gemini-1.5-flash",
    name: "Gemini 1.5 Flash",
    provider: "Google",
    requiresKey: true,
    keyEnvVar: "GOOGLE_GENERATIVE_AI_API_KEY",
    free: true,
    description: "Free - lightweight and fast",
  },
  {
    id: "groq:llama-3.1-8b-instant",
    name: "Llama 3.1 8B",
    provider: "Groq",
    requiresKey: true,
    keyEnvVar: "GROQ_API_KEY",
    free: true,
    description: "Free - fast open-source model via Groq",
  },
  {
    id: "groq:llama-3.3-70b-versatile",
    name: "Llama 3.3 70B",
    provider: "Groq",
    requiresKey: true,
    keyEnvVar: "GROQ_API_KEY",
    free: true,
    description: "Free - powerful open-source model via Groq",
  },
  {
    id: "groq:mixtral-8x7b-32768",
    name: "Mixtral 8x7B",
    provider: "Groq",
    requiresKey: true,
    keyEnvVar: "GROQ_API_KEY",
    free: true,
    description: "Free - MoE model via Groq",
  },
  {
    id: "openai:gpt-4o",
    name: "GPT-4o",
    provider: "OpenAI",
    requiresKey: true,
    keyEnvVar: "OPENAI_API_KEY",
    free: false,
    description: "Paid - most capable OpenAI model",
  },
  {
    id: "openai:gpt-4o-mini",
    name: "GPT-4o Mini",
    provider: "OpenAI",
    requiresKey: true,
    keyEnvVar: "OPENAI_API_KEY",
    free: false,
    description: "Paid - fast and affordable OpenAI model",
  },
  {
    id: "anthropic:claude-sonnet-4-20250514",
    name: "Claude Sonnet 4",
    provider: "Anthropic",
    requiresKey: true,
    keyEnvVar: "ANTHROPIC_API_KEY",
    free: false,
    description: "Paid - excellent reasoning and analysis",
  },
  {
    id: "anthropic:claude-3-5-haiku-20241022",
    name: "Claude 3.5 Haiku",
    provider: "Anthropic",
    requiresKey: true,
    keyEnvVar: "ANTHROPIC_API_KEY",
    free: false,
    description: "Paid - fast Anthropic model",
  },
];

export function getModel(id: string): ModelOption | undefined {
  return MODEL_LIST.find((m) => m.id === id);
}

export function getFreeModels(): ModelOption[] {
  return MODEL_LIST.filter((m) => m.free);
}

export function getProviders(): string[] {
  return [...new Set(MODEL_LIST.map((m) => m.provider))];
}

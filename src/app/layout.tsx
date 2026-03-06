import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "pdfGPT - Smart Document Analysis",
  description:
    "Chat with your documents using AI. Supports PDF, Word, Excel, PowerPoint. Free and paid LLM models.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-gray-100 min-h-screen antialiased">
        {children}
      </body>
    </html>
  );
}

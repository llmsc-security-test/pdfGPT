"""Agent tools for document search, page retrieval, and answer generation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from core.document import PDFProcessor
from core.vectorstore import VectorStore


@dataclass
class ToolResult:
    """Result returned by a tool execution."""
    tool_name: str
    input_value: str
    output: str
    success: bool = True


class BaseTool(ABC):
    """Abstract base class for all agent tools."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def execute(self, input_value: str) -> ToolResult:
        pass


class SearchDocumentTool(BaseTool):
    """Searches the loaded PDF document for relevant content using semantic search."""

    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self._vector_store = vector_store
        self._top_k = top_k

    @property
    def name(self) -> str:
        return "search_document"

    @property
    def description(self) -> str:
        return (
            "Search the PDF document for content relevant to a query. "
            "Input should be a search query string. "
            "Returns the most relevant passages with page numbers and relevance scores."
        )

    def execute(self, input_value: str) -> ToolResult:
        query = input_value.strip()
        if not query:
            return ToolResult(self.name, input_value, "Error: empty search query.", False)

        if not self._vector_store.is_loaded:
            return ToolResult(self.name, input_value, "Error: no document loaded.", False)

        results = self._vector_store.search(query, top_k=self._top_k)
        if not results:
            return ToolResult(
                self.name, input_value,
                "No relevant results found for this query.",
            )

        formatted = []
        for i, result in enumerate(results, 1):
            formatted.append(
                f"Result {i} [Page {result.chunk.page_number}] "
                f"(relevance: {result.score:.2f}):\n{result.chunk.text}"
            )
        output = "\n\n".join(formatted)
        return ToolResult(self.name, input_value, output)


class GetPageTool(BaseTool):
    """Retrieves the full text content of a specific page from the PDF."""

    def __init__(self, pdf_processor: PDFProcessor, pdf_path: str):
        self._processor = pdf_processor
        self._pdf_path = pdf_path

    @property
    def name(self) -> str:
        return "get_page"

    @property
    def description(self) -> str:
        return (
            "Get the full text content of a specific page from the PDF. "
            "Input should be a page number (integer). "
            "Use this when you need complete context from a specific page."
        )

    def execute(self, input_value: str) -> ToolResult:
        try:
            page_num = int(input_value.strip())
        except ValueError:
            return ToolResult(
                self.name, input_value,
                f"Error: '{input_value}' is not a valid page number.",
                False,
            )

        text = self._processor.get_page_text(self._pdf_path, page_num)
        return ToolResult(self.name, input_value, text)


class FinalAnswerTool(BaseTool):
    """Signals that the agent is ready to provide its final answer."""

    @property
    def name(self) -> str:
        return "final_answer"

    @property
    def description(self) -> str:
        return (
            "Provide your final answer to the user's question. "
            "Input should be your complete answer with [Page N] citations. "
            "Only use this when you have enough information to answer accurately."
        )

    def execute(self, input_value: str) -> ToolResult:
        return ToolResult(self.name, input_value, input_value)


class ToolRegistry:
    """Registry that holds all available tools for the agent."""

    def __init__(self):
        self._tools: dict = {}

    def register(self, tool: BaseTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def execute(self, tool_name: str, input_value: str) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name, input_value,
                f"Error: unknown tool '{tool_name}'. Available: {list(self._tools.keys())}",
                False,
            )
        return tool.execute(input_value)

    @property
    def tool_descriptions(self) -> str:
        lines = []
        for name, tool in self._tools.items():
            lines.append(f"- {name}: {tool.description}")
        return "\n".join(lines)

    @property
    def tool_names(self) -> List[str]:
        return list(self._tools.keys())

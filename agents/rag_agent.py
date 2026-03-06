"""Agentic RAG: a ReAct-style agent that plans, retrieves, reflects, and answers."""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Callable

from agents.tools import ToolRegistry
from core.llm import LLMProvider


@dataclass
class AgentStep:
    """A single step in the agent's reasoning chain."""
    step_number: int
    thought: str
    action: str
    action_input: str
    observation: str = ""

    def format_for_display(self) -> str:
        lines = [f"Step {self.step_number}:"]
        lines.append(f"  Thought: {self.thought}")
        lines.append(f"  Action: {self.action}")
        if self.action != "final_answer":
            lines.append(f"  Action Input: {self.action_input}")
        if self.observation:
            preview = self.observation[:300]
            if len(self.observation) > 300:
                preview += "..."
            lines.append(f"  Observation: {preview}")
        return "\n".join(lines)


@dataclass
class AgentResponse:
    """Complete response from the agent including reasoning trace."""
    answer: str
    steps: List[AgentStep] = field(default_factory=list)
    success: bool = True
    error_message: str = ""


SYSTEM_PROMPT = """You are an expert research assistant. A PDF document has been loaded and indexed.
Your ONLY job is to answer questions using information FROM that document.
You MUST search the document for EVERY question -- even greetings or vague queries.
Never answer from your own knowledge. Always ground your response in document content.

You have access to the following tools:
{tool_descriptions}

You MUST respond in this EXACT format for EVERY message (no exceptions):

Thought: <your reasoning about what to search for>
Action: <tool_name>
Action Input: <input for the tool>

Rules:
1. Your FIRST action for every new question MUST be search_document.
2. NEVER use final_answer without having searched at least once.
3. If initial search results are insufficient, try different search terms.
4. Use get_page when you need full context from a specific page.
5. Include [Page N] citations in your final answer for every claim.
6. If the document does not contain the requested information, state that clearly.
7. Never fabricate information not found in the document.
8. Even for greetings like "hi", search the document for a summary and respond with document info."""


class AgenticRAG:
    """ReAct-style agent that iteratively reasons, retrieves, and answers.

    The agent loop:
    1. Think about what information is needed
    2. Choose a tool and execute it
    3. Observe the result
    4. Decide whether to continue searching or provide a final answer
    5. Self-reflect on answer quality before delivering
    """

    def __init__(self, llm: LLMProvider, tools: ToolRegistry,
                 max_iterations: int = 5):
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations

    def _build_system_prompt(self) -> str:
        return SYSTEM_PROMPT.format(tool_descriptions=self._tools.tool_descriptions)

    def _build_conversation(self, question: str,
                            steps: List[AgentStep]) -> List[dict]:
        messages = [{"role": "system", "content": self._build_system_prompt()}]
        user_msg = f"Question: {question}"
        messages.append({"role": "user", "content": user_msg})

        for step in steps:
            assistant_content = (
                f"Thought: {step.thought}\n"
                f"Action: {step.action}\n"
                f"Action Input: {step.action_input}"
            )
            messages.append({"role": "assistant", "content": assistant_content})

            if step.observation:
                observation_msg = f"Observation: {step.observation}"
                messages.append({"role": "user", "content": observation_msg})

        return messages

    def _parse_agent_output(self, output: str) -> dict:
        """Parse the agent's response into thought, action, and input."""
        thought = ""
        action = ""
        action_input = ""

        thought_match = re.search(
            r"Thought:\s*(.*?)(?=\nAction:|\Z)", output, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        action_match = re.search(
            r"Action:\s*(.*?)(?=\nAction Input:|\Z)", output, re.DOTALL
        )
        if action_match:
            action = action_match.group(1).strip()

        input_match = re.search(
            r"Action Input:\s*(.*?)$", output, re.DOTALL
        )
        if input_match:
            action_input = input_match.group(1).strip()

        return {
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    def _forced_initial_search(self, question: str,
                               on_step: Optional[Callable] = None) -> AgentStep:
        """Always perform an initial document search before LLM reasoning."""
        search_query = question.strip()
        if len(search_query) < 5:
            search_query = "summary overview introduction main topic"

        tool_result = self._tools.execute("search_document", search_query)
        step = AgentStep(
            step_number=0,
            thought=f"Searching the loaded document for: {search_query}",
            action="search_document",
            action_input=search_query,
            observation=tool_result.output,
        )
        if on_step:
            on_step(step)
        return step

    def query(self, question: str,
              chat_history: Optional[List[dict]] = None,
              on_step: Optional[Callable[[AgentStep], None]] = None) -> AgentResponse:
        """Run the agentic RAG loop to answer a question.

        Args:
            question: The user's question about the document.
            chat_history: Previous conversation turns for context.
            on_step: Optional callback invoked after each reasoning step.

        Returns:
            AgentResponse with the answer and full reasoning trace.
        """
        initial_step = self._forced_initial_search(question, on_step)
        steps = [initial_step]

        for iteration in range(self._max_iterations):
            messages = self._build_conversation(question, steps)

            if chat_history:
                context_msg = "Previous conversation context:\n"
                for turn in chat_history[-6:]:
                    role = turn.get("role", "user")
                    content = turn.get("content", "")
                    context_msg += f"{role}: {content}\n"
                messages.insert(1, {"role": "user", "content": context_msg})
                messages.insert(2, {
                    "role": "assistant",
                    "content": "I have the conversation context. Let me answer the current question."
                })

            raw_output = self._llm.generate(messages)

            if raw_output.startswith("[LLM Error]"):
                return AgentResponse(
                    answer="",
                    steps=steps,
                    success=False,
                    error_message=raw_output,
                )

            parsed = self._parse_agent_output(raw_output)
            action = parsed["action"]
            action_input = parsed["action_input"]

            if not action:
                action = "final_answer"
                action_input = raw_output

            step = AgentStep(
                step_number=len(steps) + 1,
                thought=parsed["thought"],
                action=action,
                action_input=action_input,
            )

            if action == "final_answer":
                step.observation = ""
                steps.append(step)
                if on_step:
                    on_step(step)
                return AgentResponse(
                    answer=action_input,
                    steps=steps,
                    success=True,
                )

            tool_result = self._tools.execute(action, action_input)
            step.observation = tool_result.output
            steps.append(step)

            if on_step:
                on_step(step)

        fallback = self._generate_fallback_answer(question, steps)
        return AgentResponse(
            answer=fallback,
            steps=steps,
            success=True,
        )

    def _generate_fallback_answer(self, question: str,
                                  steps: List[AgentStep]) -> str:
        """Generate a final answer when the agent exhausts its iterations."""
        context_parts = []
        for step in steps:
            if step.observation and step.action != "final_answer":
                context_parts.append(step.observation)

        context = "\n\n".join(context_parts)
        prompt = (
            f"Based on the following information retrieved from a PDF document, "
            f"answer this question: {question}\n\n"
            f"Retrieved information:\n{context}\n\n"
            f"Provide a clear answer with [Page N] citations. "
            f"If the information is insufficient, state what was and was not found."
        )
        messages = [
            {"role": "system", "content": "You are a helpful research assistant."},
            {"role": "user", "content": prompt},
        ]
        return self._llm.generate(messages)

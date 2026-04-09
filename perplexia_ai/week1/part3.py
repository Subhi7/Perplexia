"""Part 3 - Conversation Memory implementation.

This implementation focuses on:
- Maintain context across messages
- Handle follow-up questions
- Use conversation history in responses
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class MemoryChat(ChatInterface):
    """Week 1 Part 3 implementation adding conversation memory."""
    
    def __init__(self):
        self.llm = None
        self.memory = None
        self.debug = True
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.tool_detection_prompt = None
        self.expression_extraction_prompt = None
        self.calculation_response_prompt = None
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for memory-enabled chat.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        - Set up conversation memory
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.memory = []

        self.tool_detection_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a routing assistant. Decide whether the current user request needs calculator usage. "
                "Use both conversation history and the current question. "
                "Reply with exactly one label: calculator or response.",
            ),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
            ),
        ])

        self.expression_extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You convert the user's request into a calculator-safe arithmetic expression. "
                "Use chat history for follow-up references like 'what about 20%'. "
                "Output only the expression using numbers, parentheses, and operators + - * /. "
                "Do not output words or explanation.",
            ),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
            ),
        ])

        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query classifier. Given conversation history and the current user question, classify the current question as exactly one of: factual, analytical, comparison, definition. "
                "Output only the category label, nothing else.",
            ),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
            ),
        ])

        self.response_prompts = {
            "factual": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a knowledgeable assistant. Use conversation context when relevant and provide a concise, direct, accurate answer.",
                ),
                (
                    "human",
                    "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
                ),
            ]),
            "analytical": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an analytical assistant. Use conversation context when relevant and provide clear step-by-step reasoning.",
                ),
                (
                    "human",
                    "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
                ),
            ]),
            "comparison": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a comparison expert. Use conversation context when relevant and provide a structured comparison with bullets or a table.",
                ),
                (
                    "human",
                    "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
                ),
            ]),
            "definition": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an educational assistant. Use conversation context when relevant and provide a clear definition with examples and use cases.",
                ),
                (
                    "human",
                    "Conversation history:\n{chat_history}\n\nCurrent user question: {question}",
                ),
            ]),
        }

        self.calculation_response_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant presenting calculator results. Use conversation history for context and answer the current question concisely.",
            ),
            (
                "human",
                "Conversation history:\n{chat_history}\n\nCurrent user question: {question}\nCalculated result: {result}",
            ),
        ])

    def _format_chat_history(self, chat_history: Optional[List[Dict[str, str]]]) -> str:
        """Convert chat history into compact text that can be injected into prompts."""
        if not chat_history:
            return "No previous conversation."

        lines: List[str] = []
        for turn in chat_history:
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def _debug_log(self, label: str, value: str) -> None:
        """Print debug details when debug mode is enabled."""
        if self.debug:
            print(f"[Part3 Debug] {label}: {value}")
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with memory and tools.
        
        Students should:
        - Use chat history for context
        - Handle follow-up questions
        - Use calculator when needed
        - Format responses appropriately
        
        Args:
            message: The user's input message
            chat_history: List of previous chat messages
            
        Returns:
            str: The assistant's response
        """
        parser = StrOutputParser()
        history_text = self._format_chat_history(chat_history)
        self.memory = chat_history or []
        self._debug_log("message", message)
        self._debug_log("history_text", history_text)

        # Step 1: Decide whether this turn should use the calculator.
        tool_router_chain = self.tool_detection_prompt | self.llm | parser
        route = tool_router_chain.invoke({"question": message, "chat_history": history_text}).strip().lower()
        self._debug_log("route", route)

        if route == "calculator":
            # Step 2a: Build expression from current question + history.
            expression_chain = self.expression_extraction_prompt | self.llm | parser
            expression = expression_chain.invoke({"question": message, "chat_history": history_text}).strip()
            self._debug_log("expression", expression)

            # Step 2b: Evaluate using the provided calculator tool.
            result = self.calculator.evaluate_expression(expression)
            self._debug_log("calculator_result", str(result))
            if isinstance(result, str) and result.startswith("Error"):
                return result

            # Step 2c: Convert raw tool output to a user-facing response.
            calculator_response_chain = self.calculation_response_prompt | self.llm | parser
            return calculator_response_chain.invoke(
                {"question": message, "result": result, "chat_history": history_text}
            )

        # Step 3: Non-tool queries follow classification + style prompts.
        classifier_chain = self.query_classifier_prompt | self.llm | parser
        category = classifier_chain.invoke({"question": message, "chat_history": history_text}).strip().lower()
        self._debug_log("category", category)

        if category not in self.response_prompts:
            category = "factual"

        response_chain = self.response_prompts[category] | self.llm | parser
        return response_chain.invoke({"question": message, "chat_history": history_text})

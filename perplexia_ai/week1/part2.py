"""Part 2 - Basic Tools implementation.

This implementation focuses on:
- Detect when calculations are needed
- Use calculator for mathematical operations
- Format calculation results clearly
"""

from typing import Dict, List, Optional

from perplexia_ai.core.chat_interface import ChatInterface
from perplexia_ai.tools.calculator import Calculator
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class BasicToolsChat(ChatInterface):
    """Week 1 Part 2 implementation adding calculator functionality."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
        self.tool_detection_prompt = None
        self.expression_extraction_prompt = None
        self.calculation_response_prompt = None
        self.calculator = Calculator()
    
    def initialize(self) -> None:
        """Initialize components for basic tools.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        - Initialize calculator tool
        """
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        self.tool_detection_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a routing assistant. Determine whether the user's message requires a calculator. "
                "Reply with exactly one label: calculator or response. Use calculator only when the user is asking "
                "for a numeric computation, arithmetic operation, percentage, total, tip, discount, or similar math task. "
                "Otherwise reply with response.",
            ),
            ("human", "{question}"),
        ])

        self.expression_extraction_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You extract a valid arithmetic expression from the user's request for a calculator tool. "
                "Return only the arithmetic expression using numbers, parentheses, and operators +, -, *, /. "
                "Do not include units, words, explanation, or formatting.",
            ),
            ("human", "{question}"),
        ])

        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a query classifier. Given a user question, classify it as exactly one of: factual, analytical, comparison, definition. "
                "Output only the category label, nothing else.",
            ),
            ("human", "{question}"),
        ])

        self.response_prompts = {
            "factual": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a knowledgeable assistant. Provide a concise, direct, and accurate answer to the user's factual question. Keep it brief and to the point.",
                ),
                ("human", "{question}"),
            ]),
            "analytical": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an analytical assistant. Provide a thorough response that includes step-by-step reasoning. Break down your explanation into clear logical steps.",
                ),
                ("human", "{question}"),
            ]),
            "comparison": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are a comparison expert. Provide a well-structured comparison using bullet points or a table format. Highlight key similarities and differences clearly.",
                ),
                ("human", "{question}"),
            ]),
            "definition": ChatPromptTemplate.from_messages([
                (
                    "system",
                    "You are an educational assistant. Provide a clear definition followed by practical examples and common use cases. Make it easy to understand.",
                ),
                ("human", "{question}"),
            ]),
        }

        self.calculation_response_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant presenting calculator results. Use the original user request and the computed result to produce a natural, concise answer. "
                "If useful, mention both the computed value and any relevant context from the request.",
            ),
            (
                "human",
                "User request: {question}\nCalculated result: {result}",
            ),
        ])
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message with calculator support.
        
        Students should:
        - Check if calculation needed
        - Use calculator if needed
        - Otherwise, handle as regular query
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 2
            
        Returns:
            str: The assistant's response
        """
        parser = StrOutputParser()

        # Step 1: Route between calculator flow and normal response flow.
        tool_router_chain = self.tool_detection_prompt | self.llm | parser
        route = tool_router_chain.invoke({"question": message}).strip().lower()

        if route == "calculator":
            # Step 2a: Convert the user request to an arithmetic expression.
            expression_chain = self.expression_extraction_prompt | self.llm | parser
            expression = expression_chain.invoke({"question": message}).strip()

            # Step 2b: Execute the expression with the provided Calculator tool.
            result = self.calculator.evaluate_expression(expression)

            # Return calculator errors directly to keep behavior explicit.
            if isinstance(result, str) and result.startswith("Error"):
                return result

            # Step 2c: Format tool output into a natural language response.
            calculator_response_chain = self.calculation_response_prompt | self.llm | parser
            return calculator_response_chain.invoke({"question": message, "result": result})

        # Step 3: Fallback to the Part 1 query-understanding flow.
        classifier_chain = self.query_classifier_prompt | self.llm | parser
        category = classifier_chain.invoke({"question": message}).strip().lower()

        if category not in self.response_prompts:
            category = "factual"

        response_chain = self.response_prompts[category] | self.llm | parser
        return response_chain.invoke({"question": message})
"""Part 1 - Query Understanding implementation.

This implementation focuses on:
- Classify different types of questions
- Format responses based on query type
- Present information professionally
"""

from typing import Dict, List, Optional
from perplexia_ai.core.chat_interface import ChatInterface
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class QueryUnderstandingChat(ChatInterface):
    """Week 1 Part 1 implementation focusing on query understanding."""
    
    def __init__(self):
        self.llm = None
        self.query_classifier_prompt = None
        self.response_prompts = {}
    
    def initialize(self) -> None:
        """Initialize components for query understanding.
        
        Students should:
        - Initialize the chat model
        - Set up query classification prompts
        - Set up response formatting prompts
        """
        self.query_classifier_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a query classifier. Given a user question, classify it as exactly one of: factual, analytical, comparison, definition. Output only the category label, nothing else."),
            ("human", "{question}")
            ])
        
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        self.response_prompts = {
            "factual": ChatPromptTemplate.from_messages([
                ("system", "You are a knowledgeable assistant. Provide a concise, direct, and accurate answer to the user's factual question. Keep it brief and to the point."),
                ("human", "{question}")
            ]),
            "analytical": ChatPromptTemplate.from_messages([
                ("system", "You are an analytical assistant. Provide a thorough response that includes step-by-step reasoning. Break down your explanation into clear logical steps."),
                ("human", "{question}")
            ]),
            "comparison": ChatPromptTemplate.from_messages([
                ("system", "You are a comparison expert. Provide a well-structured comparison using bullet points or a table format. Highlight key similarities and differences clearly."),
                ("human", "{question}")
            ]),
            "definition": ChatPromptTemplate.from_messages([
                ("system", "You are an educational assistant. Provide a clear definition followed by practical examples and common use cases. Make it easy to understand."),
                ("human", "{question}")
            ]),
        }
    
    def process_message(self, message: str, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Process a message using query understanding.
        
        Students should:
        - Classify the query type
        - Generate appropriate response
        - Format based on query type
        
        Args:
            message: The user's input message
            chat_history: Not used in Part 1
            
        Returns:
            str: The assistant's response
        """
        # Step 1: Classify the query
        # Chain: prompt → LLM → parse to string
        classifier_chain = self.query_classifier_prompt | self.llm | StrOutputParser()
        category = classifier_chain.invoke({"question": message}).strip().lower()
        
        # Step 2: Pick the right response prompt using dictionary lookup
        # If classifier returns something unexpected, default to "factual"
        if category not in self.response_prompts:
            category = "factual"
        response_prompt = self.response_prompts[category]
        
        # Step 3: Generate the response using the selected prompt
        response_chain = response_prompt | self.llm | StrOutputParser()
        response = response_chain.invoke({"question": message})
        
        return response
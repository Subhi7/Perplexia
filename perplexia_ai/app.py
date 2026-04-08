import gradio as gr
from typing import Dict, List
from dotenv import load_dotenv

from perplexia_ai.week1.factory import Week1Mode, create_chat_implementation

# Load environment variables
load_dotenv()

APP_THEME = gr.themes.Soft()


def create_demo(mode_str: str = "part1") -> gr.ChatInterface:
    """Create and return a Gradio demo with the specified mode.

    Args:
        mode_str: String representation of the mode ('part1', 'part2', or 'part3')

    Returns:
        gr.ChatInterface: Configured Gradio chat interface
    """
    mode_map = {
        "part1": Week1Mode.PART1_QUERY_UNDERSTANDING,
        "part2": Week1Mode.PART2_BASIC_TOOLS,
        "part3": Week1Mode.PART3_MEMORY,
    }

    if mode_str not in mode_map:
        raise ValueError(f"Unknown mode: {mode_str}. Choose from: {list(mode_map.keys())}")

    mode = mode_map[mode_str]

    chat_interface = create_chat_implementation(mode)
    chat_interface.initialize()

    def respond(message: str, history: List[Dict[str, str]]) -> str:
        """Process the message and return a response."""
        return chat_interface.process_message(message, history)

    titles = {
        "part1": "Perplexia AI - Query Understanding",
        "part2": "Perplexia AI - Basic Tools",
        "part3": "Perplexia AI - Memory",
    }

    descriptions = {
        "part1": "Your intelligent AI assistant that can understand different types of questions and format responses accordingly.",
        "part2": "Your intelligent AI assistant that can answer questions, perform calculations, and format responses.",
        "part3": "Your intelligent AI assistant that can answer questions, perform calculations, and maintain conversation context.",
    }

    demo = gr.ChatInterface(
        fn=respond,
        title=titles[mode_str],
        description=descriptions[mode_str],
        examples=[
            "What is machine learning?",
            "Compare SQL and NoSQL databases",
            "If I have a dinner bill of $120, what would be a 15% tip?",
            "What about 20%?",
        ],
        run_examples_on_click=True,
        chatbot=gr.Chatbot(show_label=False, height=560, layout="bubble"),
        textbox=gr.Textbox(
            show_label=False,
            placeholder="Ask a question, compare ideas, or run a quick calculation...",
            lines=1,
            max_lines=5,
            submit_btn="Send",
            stop_btn="Stop",
        ),
        fill_height=True,
        fill_width=True,
        show_progress="minimal",
    )
    return demo

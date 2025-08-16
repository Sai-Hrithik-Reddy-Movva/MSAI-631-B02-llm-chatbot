import gradio as gr
from transformers import pipeline, Conversation
import os # Import os to check for environment variables for model path

# Define a default model. You can change this.
DEFAULT_MODEL = "microsoft/DialoGPT-medium"
FALLBACK_MODEL = "facebook/blenderbot-400M-distill" # A smaller, more accessible model

# Attempt to load a model. Prioritize a larger model if possible, fallback if not.
try:
    print(f"Attempting to load model: {DEFAULT_MODEL}")
    # We explicitly specify device="cpu" to avoid CUDA errors if no GPU is available
    # or if PyTorch/Transformers isn't set up for GPU correctly.
    # For GPU usage, remove device="cpu" or set it to device=0 (for GPU index 0).
    chatbot_pipeline = pipeline("conversational", model=DEFAULT_MODEL, device="cpu")
    print(f"Successfully loaded {DEFAULT_MODEL}")
except Exception as e:
    print(f"Error loading {DEFAULT_MODEL}: {e}")
    print(f"Trying a smaller fallback model for demonstration: {FALLBACK_MODEL}")
    try:
        chatbot_pipeline = pipeline("conversational", model=FALLBACK_MODEL, device="cpu")
        print(f"Successfully loaded {FALLBACK_MODEL}")
    except Exception as e:
        print(f"Error loading {FALLBACK_MODEL}: {e}")
        print("Could not load any conversational model. Please check your internet connection,")
        print("Hugging Face Hub access, and ensure your `transformers` library is updated.")
        print("You might also need to install PyTorch/TensorFlow correctly.")
        # Exit or raise error if no model can be loaded
        exit("Failed to load any chatbot model. Exiting.")


def chatbot_response(message, history):
    """
    This function takes the user's message and the conversation history,
    then uses the Hugging Face pipeline to generate a response.
    """
    # Gradio's history is a list of lists: [[user_msg_1, bot_msg_1], [user_msg_2, bot_msg_2], ...]
    # Hugging Face's Conversation object needs the turns in sequence.

    # Initialize a new Conversation object
    conversation = Conversation()

    # Add past turns to the Conversation object
    for user_msg, bot_msg in history:
        conversation.add_user_input(user_msg)
        conversation.append_response(bot_msg)

    # Add the current user input
    conversation.add_user_input(message)

    # Get the model's response
    # The pipeline automatically updates the conversation object
    response_obj = chatbot_pipeline(conversation)

    # The last generated response is what we need
    bot_message = response_obj.generated_responses[-1]

    return bot_message

# Create the Gradio interface
demo = gr.ChatInterface(
    fn=chatbot_response,
    title="Hugging Face Chatbot with Gradio 🤖",
    description="Ask me anything! I'm powered by a Hugging Face conversational model.",
    examples=[["Hello!"], ["What is AI?"], ["Tell me a joke."]],
    theme="soft", # You can choose different themes like "default", "grass", "huggingface"
    # Optional: Set share=True to get a public link (for temporary demos)
    # share=True
)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()
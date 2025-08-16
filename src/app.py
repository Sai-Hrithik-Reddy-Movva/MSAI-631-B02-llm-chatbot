import gradio as gr
from transformers import AutoModelForCausalLM, GPT2Tokenizer
import torch

# Import the RAG module
from rag import create_and_populate_vector_store, retrieve_context, collection_name

# This script implements a lightweight conversational chatbot based on the provided design document.
# It uses Gradio for the front-end interface and the Hugging Face Transformers library for the
# language model back-end. The chosen model, 'distilgpt2', is a small and efficient
# causal language model that is suitable for CPU-only deployment, as specified in the document.

# ----------------------------------------------------------------------------------
# 1. Model and Tokenizer Initialization
# ----------------------------------------------------------------------------------
# Load the pre-trained model and tokenizer.
# The 'distilgpt2' model is a distilled version of GPT-2, making it smaller and faster.
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Set the pad token for the tokenizer and model. This is important for batch processing
# and helps the model handle varying input lengths.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

# ----------------------------------------------------------------------------------
# 2. RAG Setup
# ----------------------------------------------------------------------------------
# We will create and populate the vector store with sample documents when the app starts.
# In a real-world scenario, you would do this as a separate data preparation step.
# For demonstration purposes, we'll keep the sample documents here.
sample_documents = [
    "Quicksort is an efficient, in-place, and unstable sorting algorithm. It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.",
    "Python code for Quicksort:\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n        less = [i for i in arr[1:] if i <= pivot]\n        greater = [i for i in arr[1:] if i > pivot]\n        return quicksort(less) + [pivot] + quicksort(greater)",
    "Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted. The algorithm gets its name from the way smaller elements 'bubble' to the top of the list.",
    "A stack is a linear data structure which follows a particular order in which the operations are performed. The order may be LIFO (Last In First Out) or FILO (First In Last Out). The primary operations on a stack are Push (add an item) and Pop (remove an item).",
    "A queue is a linear data structure which follows a particular order in which the operations are performed. The order is FIFO (First In First Out). The primary operations on a queue are Enqueue (add an item) and Dequeue (remove an item)."
]

# Run this once to set up the knowledge base for the RAG system.
create_and_populate_vector_store(sample_documents, collection_name)

# ----------------------------------------------------------------------------------
# 3. Chatbot Logic Function
# ----------------------------------------------------------------------------------
# This function is the core of the chatbot. It processes user input, manages conversation
# history, and generates a response from the language model.

def generate_response(message: str, history: list) -> str:
    """
    Generates a response from the DistilGPT2 model with RAG.

    Args:
        message (str): The new message from the user.
        history (list): A list of tuples containing the conversation history.
                        Each tuple is of the form (user_message, chatbot_response).

    Returns:
        str: The generated response from the chatbot.
    """
    # Use the RAG module to retrieve relevant context based on the user's message.
    retrieved_documents = retrieve_context(message, collection_name, n_results=2)
    
    # Prepend the retrieved context to the user's message to "ground" the LLM's response.
    # The design document specifies this approach.
    context_prefix = "\n".join(retrieved_documents)
    
    # Concatenate the chat history with the retrieved context and the new message.
    full_prompt = ""
    if context_prefix:
        full_prompt += f"Context:\n{context_prefix}\n\n"
    
    for user_msg, bot_msg in history:
        full_prompt += f"User: {user_msg}\nBot: {bot_msg}\n"
    
    # Add the latest user message to the prompt.
    full_prompt += f"User: {message}\nBot: "

    # Encode the prompt text into model-understandable tensors.
    inputs = tokenizer.encode(full_prompt, return_tensors="pt")

    # Generate a response from the model.
    outputs = model.generate(
        inputs, 
        max_new_tokens=100, 
        pad_token_id=model.config.eos_token_id,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode the generated tokens back into a string.
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the chatbot's part of the response.
    start_index = response.rfind("Bot:")
    if start_index != -1:
        bot_response = response[start_index + 4:].strip()
    else:
        bot_response = response.strip()

    return bot_response

# ----------------------------------------------------------------------------------
# 4. Gradio Interface Setup
# ----------------------------------------------------------------------------------
# Use Gradio's `ChatInterface` to create a simple and intuitive UI.
chatbot_interface = gr.ChatInterface(
    fn=generate_response,
    title="Lightweight LLM Chatbot with RAG",
    description="A conversational agent for computer science learners, now with Retrieval-Augmented Generation.",
    submit_btn="Send",
    # retry_btn="Try Again",
    # undo_btn="Delete Last",
    # clear_btn="Clear History",
)

# Launch the Gradio application.
if __name__ == "__main__":
    chatbot_interface.launch(debug=True)

# Lightweight LLM Chatbot

This project implements a lightweight conversational chatbot designed to assist computer science students. Built with Gradio for the front-end and Hugging Face Transformers for the back-end, it is designed for deployability on regular laptops and Hugging Face Spaces.

## Features

- **Gradio Interface:** A simple, user-friendly chat interface.
- **Lightweight Model:** Uses `distilgpt2` for efficient, CPU-friendly performance.
- **Conversation History:** Maintains a short history to provide context.
- **Open Source:** Built with open-source tools to ensure reproducibility.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/Sai-Hrithik-Reddy-Movva/MSAI-631-B02-llm-chatbot.git]
    cd chatbot
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To start the chatbot, run the main application script:

```bash
python src/app.py
```

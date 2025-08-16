import unittest
import os
import shutil

# Import the core functions from your source files
from app import generate_response
from rag import create_and_populate_vector_store, retrieve_context, collection_name

# This test file uses the built-in `unittest` framework to check the functionality
# of the RAG module and the main chatbot logic.
# The tests are designed to be simple and verify that the functions
# behave correctly under typical conditions.

class TestChatbot(unittest.TestCase):
    """
    Test suite for the chatbot's core components.
    """

    def setUp(self):
        """
        Set up the testing environment before each test.
        This includes creating a temporary ChromaDB instance with a small set of documents.
        """
        # Define a temporary directory for the test database to avoid polluting the main one.
        self.test_db_dir = "./test_chroma_db"
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
        
        # Override the collection name for testing
        self.test_collection_name = "test_cs_learning_docs"
        
        # Sample documents for testing the retrieval function
        self.sample_documents = [
            "Quicksort is an efficient sorting algorithm.",
            "Python code for Quicksort is often used in interviews.",
            "A stack is a linear data structure.",
            "A queue is another linear data structure."
        ]
        
        # Create and populate a temporary vector store for the tests
        print("Setting up test database...")
        create_and_populate_vector_store(self.sample_documents, self.test_collection_name)
        print("Setup complete.")

    def tearDown(self):
        """
        Clean up the testing environment after each test.
        This removes the temporary test database.
        """
        print("Cleaning up test database...")
        if os.path.exists(self.test_db_dir):
            shutil.rmtree(self.test_db_dir)
        print("Cleanup complete.")

    def test_retrieve_context_returns_list(self):
        """
        Test that the retrieve_context function returns a list.
        """
        query = "What is quicksort?"
        context = retrieve_context(query, self.test_collection_name)
        self.assertIsInstance(context, list, "The retrieved context should be a list.")
        
    def test_retrieve_context_finds_relevant_docs(self):
        """
        Test that a query for 'quicksort' retrieves documents containing the word 'quicksort'.
        """
        query = "How does quicksort work?"
        context = retrieve_context(query, self.test_collection_name)
        
        # Check if the most relevant documents are returned
        self.assertTrue(any("Quicksort" in doc for doc in context),
                        "The retrieved context should contain documents about quicksort.")

    def test_generate_response_no_history(self):
        """
        Test that the generate_response function works with no history.
        """
        # Use a mock message for a simple test
        message = "Tell me about a stack data structure."
        response = generate_response(message, [])
        self.assertIsInstance(response, str, "The response should be a string.")
        self.assertGreater(len(response), 0, "The response should not be empty.")

    def test_generate_response_with_history(self):
        """
        Test that the generate_response function works with a conversation history.
        """
        history = [("Hello", "Hi there!")]
        message = "What about a queue?"
        response = generate_response(message, history)
        self.assertIsInstance(response, str, "The response should be a string.")
        self.assertGreater(len(response), 0, "The response should not be empty.")

# To run the tests from the command line, use:
# python -m unittest tests.test_basic.py
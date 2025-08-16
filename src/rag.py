# Import necessary libraries for RAG
import chromadb
from sentence_transformers import SentenceTransformer
import os

# This script implements the Retrieval-Augmented Generation (RAG) module.
# Its primary purpose is to retrieve relevant information from a predefined set of documents
# to "ground" the chatbot's responses and prevent factual inaccuracies (hallucinations).
# The design document specifies using ChromaDB and prepending retrieved passages to the prompt.

# ----------------------------------------------------------------------------------
# 1. Initialization and Configuration
# ----------------------------------------------------------------------------------
# The `SentenceTransformer` model is a lightweight, general-purpose model for
# creating embeddings (numerical representations of text).
# 'all-MiniLM-L6-v2' is an excellent, compact choice for this purpose.
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the ChromaDB client. The code uses a persistent client to save data
# to a local directory, making it reusable between sessions.
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection_name = "cs_learning_docs"

# ----------------------------------------------------------------------------------
# 2. Document Processing and Vector Store Creation
# ----------------------------------------------------------------------------------
# The following functions handle creating and populating the vector store.

def create_and_populate_vector_store(documents: list, collection: str):
    """
    Creates a new ChromaDB collection and populates it with documents.
    This process involves embedding the documents and adding them to the collection.
    
    Args:
        documents (list): A list of strings, where each string is a document.
        collection (str): The name of the collection to create.
    """
    print(f"Creating or getting collection: {collection}")
    
    try:
        # Get or create the collection.
        # Use the SentenceTransformer's encode method directly as the embedding function
        # This is a more direct way of using it with ChromaDB.
        db_collection = chroma_client.get_or_create_collection(
            name=collection,
            embedding_function=lambda texts: embedding_model.encode(texts).tolist()
        )
        
        # Add the documents to the collection.
        db_collection.add(
            documents=documents,
            ids=[f"doc_{i}" for i in range(len(documents))]
        )
        print(f"Successfully added {len(documents)} documents to the collection.")
    except Exception as e:
        print(f"An error occurred while populating the vector store: {e}")

# ----------------------------------------------------------------------------------
# 3. Context Retrieval
# ----------------------------------------------------------------------------------

def retrieve_context(query: str, collection: str, n_results: int = 3) -> list:
    """
    Retrieves the most relevant documents from the vector store based on a query.
    
    Args:
        query (str): The user's natural language query.
        collection (str): The name of the collection to search.
        n_results (int): The number of top-k results to return.
        
    Returns:
        list: A list of the most relevant documents (strings).
    """
    try:
        db_collection = chroma_client.get_collection(
            name=collection,
            embedding_function=lambda texts: embedding_model.encode(texts).tolist()
        )
        
        results = db_collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved_documents = results.get("documents", [[]])[0]
        
        if not retrieved_documents:
            print("No relevant documents found.")
            return []
            
        print(f"Retrieved {len(retrieved_documents)} documents.")
        return retrieved_documents
        
    except Exception as e:
        print(f"An error occurred while retrieving context: {e}")
        return []

# ----------------------------------------------------------------------------------
# 4. Example Usage (for testing)
# ----------------------------------------------------------------------------------

if __name__ == '__main__':
    sample_documents = [
        "Quicksort is an efficient, in-place, and unstable sorting algorithm. It works by selecting a 'pivot' element from the array and partitioning the other elements into two sub-arrays, according to whether they are less than or greater than the pivot. The sub-arrays are then sorted recursively.",
        "Python code for Quicksort:\n\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    else:\n        pivot = arr[0]\n        less = [i for i in arr[1:] if i <= pivot]\n        greater = [i for i in arr[1:] if i > pivot]\n        return quicksort(less) + [pivot] + quicksort(greater)",
        "Bubble Sort is a simple sorting algorithm that repeatedly steps through the list, compares adjacent elements and swaps them if they are in the wrong order. The pass through the list is repeated until the list is sorted. The algorithm gets its name from the way smaller elements 'bubble' to the top of the list.",
        "A stack is a linear data structure which follows a particular order in which the operations are performed. The order may be LIFO (Last In First Out) or FILO (First In Last Out). The primary operations on a stack are Push (add an item) and Pop (remove an item).",
        "A queue is a linear data structure which follows a particular order in which the operations are performed. The order is FIFO (First In First Out). The primary operations on a queue are Enqueue (add an item) and Dequeue (remove an item)."
    ]
    
    create_and_populate_vector_store(sample_documents, collection_name)
    
    user_query = "Can you explain how quicksort works and show me the Python code?"
    
    retrieved_context = retrieve_context(user_query, collection_name, n_results=2)
    
    print("\n--- Retrieved Context for Prompting ---")
    if retrieved_context:
        for i, doc in enumerate(retrieved_context):
            print(f"Passage {i+1}:\n{doc}\n")
    else:
        print("No context was retrieved.")

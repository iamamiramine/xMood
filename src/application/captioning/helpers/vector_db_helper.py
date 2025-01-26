import os
from typing import List

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from time import sleep

EMBED_DELAY = 0.02  # 20 milliseconds


class EmbeddingProxy:
    """
    A proxy class that wraps an embedding model to add rate limiting delays between embedding calls.
    This helps prevent overloading the embedding model with too many rapid requests.
    """
    def __init__(self, embedding):
        """
        Initialize the proxy with an embedding model.
        
        Args:
            embedding: The embedding model to wrap
        """
        self.embedding = embedding

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of texts with a delay between calls.
        
        Args:
            texts: List of strings to embed
            
        Returns:
            List of embedding vectors (list of floats) for each input text
        """
        sleep(EMBED_DELAY)  # Add delay before embedding
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text with a delay between calls.
        
        Args:
            text: String to embed
            
        Returns:
            Embedding vector (list of floats) for the input text
        """
        sleep(EMBED_DELAY)  # Add delay before embedding
        return self.embedding.embed_query(text)



def create_vector_db(texts, embeddings=None, collection_name="chroma", force_reload=False):
    """
    Create or load a vector database for document storage and retrieval.
    
    Args:
        texts: List of documents to add to the database
        embeddings: Optional embedding model to use (defaults to HuggingFace all-mpnet-base-v2)
        collection_name: Name for the Chroma collection (defaults to "chroma")
        force_reload: Whether to force recreating the database even if it exists
        
    Returns:
        Chroma: The vector database instance
        
    Raises:
        Exception: If there is an error creating or loading the database
    """
    try:
        # Initialize embeddings model if not provided
        if not embeddings:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Wrap embeddings with rate limiting proxy
        proxy_embeddings = EmbeddingProxy(embeddings)
        persist_directory = os.path.join("./data/store/", collection_name)
        
        # Load existing database if it exists and force_reload is False
        if os.path.exists(persist_directory) and not force_reload:
            print(f"Loading existing vector database from {persist_directory}", flush=True)
            db = Chroma(
                collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=persist_directory
            )
        else:
            # Handle empty texts input
            if not texts:
                print("Empty texts passed in to create vector database", flush=True)
                texts = []
            
            # Remove existing database directory if force_reload
            if force_reload and os.path.exists(persist_directory):
                import shutil
                shutil.rmtree(persist_directory)
            
            # Create new vector database
            print(f"Creating new vector database in {persist_directory}", flush=True)
            db = Chroma(
                collection_name=collection_name,
                embedding_function=proxy_embeddings,
                persist_directory=persist_directory
            )
            
            # Add documents if provided
            if texts:
                print(f"Processing {len(texts)} documents", flush=True)
                # Filter for valid documents with non-empty content
                valid_texts = [
                    text for text in texts 
                    if hasattr(text, 'page_content') and text.page_content.strip()
                ]
                print(f"Found {len(valid_texts)} valid documents", flush=True)
                db.add_documents(valid_texts)

        return db
    except Exception as e:
        print(f"Error creating vector database: {e}", flush=True)
        raise


def create_hyde_vector_store(texts, embeddings, hypothetical_embeddings, collection_name="hyde_store"):
    """
    Creates a vector store for Hypothetical Document Embeddings (HYDE) approach.
    
    Args:
        texts: List of source documents to embed
        embeddings: Embedding function to convert texts to vectors
        hypothetical_embeddings: List of hypothetical documents to embed
        collection_name: Name of the Chroma collection (default: "hyde_store")
        
    Returns:
        Chroma: Vector store containing both source and hypothetical document embeddings
    """
    # Initialize Chroma vector store with embedding function
    db = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings
    )
    
    # Add source documents if provided
    if texts:
        db.add_documents(texts)
    
    # Add hypothetical documents if provided    
    if hypothetical_embeddings:
        db.add_documents(hypothetical_embeddings)
        
    return db

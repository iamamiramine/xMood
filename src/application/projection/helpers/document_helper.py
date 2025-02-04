import os
from pathlib import Path
import pickle
import hashlib
from typing import Dict

from pypdf import PdfReader

from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_core.messages.base import BaseMessage


def split_documents(docs):
    """
    Split documents into smaller chunks for processing.

    Args:
        docs: List of documents to split. Can be either raw text content
             or Document objects with page_content attribute.

    Returns:
        list: List of Document objects containing the chunked text.

    The function uses RecursiveCharacterTextSplitter to split documents into
    chunks of 1000 characters with no overlap between chunks. This helps
    maintain reasonable context sizes for embedding and retrieval.
    """
    # Initialize text splitter with desired chunk parameters
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=0, length_function=len, is_separator_regex=False  # Each chunk will be ~1000 chars  # No overlap between chunks
    )

    # Extract raw text content if docs are Document objects
    contents = docs
    if docs and isinstance(docs[0], Document):
        contents = [doc.page_content for doc in docs]

    # Split contents into chunks and create Document objects
    texts = text_splitter.create_documents(contents)
    n_chunks = len(texts)
    print(f"Split into {n_chunks} chunks")
    return texts


def get_files_hash(data_dir: str = "./data") -> str:
    """
    Generate a hash value representing the state of all text, CSV and PDF files in a directory.

    Args:
        data_dir (str): Directory path to scan for files. Defaults to "./data"

    Returns:
        str: MD5 hash digest of the combined file information

    The function:
    1. Recursively finds all .txt, .csv and .pdf files in the directory
    2. For each file, collects path, size and modification time
    3. Sorts the file information for consistent hashing
    4. Generates an MD5 hash of the combined file information

    This hash can be used to detect when files have changed.
    """
    # List to store tuples of (path, size, mtime) for each file
    files_info = []

    # Collect info for all .txt files
    for path in Path(data_dir).glob("**/*.txt"):
        stat = path.stat()
        files_info.append((str(path), stat.st_size, stat.st_mtime))

    # Collect info for all .csv files
    for path in Path(data_dir).glob("**/*.csv"):
        stat = path.stat()
        files_info.append((str(path), stat.st_size, stat.st_mtime))

    # Collect info for all .pdf files
    for path in Path(data_dir).glob("**/*.pdf"):
        stat = path.stat()
        files_info.append((str(path), stat.st_size, stat.st_mtime))

    # Sort for consistent hashing
    files_info.sort()

    # Generate MD5 hash of all file information
    hasher = hashlib.md5()
    for file_info in files_info:
        hasher.update(str(file_info).encode())

    return hasher.hexdigest()


def save_dataset_metadata(hash_value: str, file_path: str = "./data/store/dataset_metadata.pkl"):
    """
    Save dataset metadata including hash and timestamp to a pickle file.

    Args:
        hash_value (str): Hash value representing the current state of the dataset
        file_path (str): Path where metadata will be saved. Defaults to "./data/store/dataset_metadata.pkl"
    """
    # Create directories in path if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create metadata dictionary with hash and timestamp
    metadata = {
        "hash": hash_value,
        # Get file modification time if file exists, otherwise None
        "timestamp": os.path.getmtime(file_path) if os.path.exists(file_path) else None,
    }

    # Save metadata to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(metadata, f)


def load_dataset_metadata(file_path: str = "./data/store/dataset_metadata.pkl") -> Dict:
    """
    Load dataset metadata from a pickle file.

    Args:
        file_path (str): Path to the metadata pickle file.
                        Defaults to "./data/store/dataset_metadata.pkl"

    Returns:
        Dict: Dictionary containing:
            - hash (str or None): Hash value representing dataset state
            - timestamp (float or None): Last modification timestamp
    """
    # Return default metadata if file doesn't exist
    if not os.path.exists(file_path):
        return {"hash": None, "timestamp": None}

    # Load and return metadata from pickle file
    with open(file_path, "rb") as f:
        return pickle.load(f)


def has_dataset_changed(data_dir: str = "./data") -> bool:
    """
    Check if the dataset has changed by comparing current and stored hash values.

    Args:
        data_dir (str): Directory containing the dataset files. Defaults to "./data"

    Returns:
        bool: True if dataset has changed (hashes don't match), False otherwise
    """
    # Get hash of current dataset files
    current_hash = get_files_hash(data_dir)

    # Load previously stored metadata containing old hash
    stored_metadata = load_dataset_metadata()

    # Compare current and stored hashes to detect changes
    return current_hash != stored_metadata["hash"]


def save_text_chunks(texts, file_path: str = "./data/store/chunks.pkl"):
    """
    Save text chunks to a pickle file and update dataset metadata.

    Args:
        texts: List of text chunks to save
        file_path (str): Path where chunks will be saved. Defaults to "./data/store/chunks.pkl"
    """
    # Create directories in path if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Save text chunks to pickle file
    with open(file_path, "wb") as f:
        pickle.dump(texts, f)

    # Calculate and save hash of current dataset state
    current_hash = get_files_hash("./data")
    save_dataset_metadata(current_hash)


def load_txt_files(data_dir: str = "./data", force_reload: bool = False):
    # Check if we can use cached chunks
    if not force_reload and not has_dataset_changed(data_dir):
        # Try to load existing chunks if dataset hasn't changed
        chunks_path = "./data/store/chunks.pkl"
        if os.path.exists(chunks_path):
            print("Loading existing chunks (dataset unchanged)")
            # Load and return cached chunks
            return load_text_chunks(chunks_path)

    # Dataset has changed or force reload requested
    print("Processing files (dataset changed or force reload)", flush=True)

    # Initialize empty list to store loaded documents
    docs = []

    # Get list of all .txt files in data directory
    paths = list_txt_files(data_dir)

    # Load each text file
    for path in paths:
        print(f"Loading {path}", flush=True)
        # Create TextLoader for current file
        loader = TextLoader(path)
        # Load document and add to docs list
        docs.extend(loader.load())

    # Split loaded documents into chunks
    texts = split_documents(docs)

    # Cache the chunks for future use
    save_text_chunks(texts)

    return texts


def load_text_chunks(file_path):
    """
    Load text chunks from a pickle file.

    Args:
        file_path (str): Path to the pickle file containing text chunks

    Returns:
        list: The loaded text chunks from the pickle file
    """
    # Open file in binary read mode
    with open(file_path, "rb") as f:
        # Load and return the pickled text chunks
        return pickle.load(f)


def get_question(input):
    """
    Extract the question text from various input formats.

    Args:
        input: The input to extract the question from. Can be:
            - None: Returns None
            - str: Returns the string directly
            - dict: Returns the value of the 'question' key
            - BaseMessage: Returns the message content

    Returns:
        str or None: The extracted question text

    Raises:
        Exception: If input format is not supported
    """
    # Return None for empty input
    if not input:
        return None
    # Return string input directly
    elif isinstance(input, str):
        return input
    # Extract question from dict if it has 'question' key
    elif isinstance(input, dict) and "question" in input:
        return input["question"]
    # Get content from BaseMessage objects
    elif isinstance(input, BaseMessage):
        return input.content
    # Raise error for unsupported input types
    else:
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def format_docs(docs):
    """
    Format and sanitize a list of document objects into a single string.

    Args:
        docs: List of document objects that have a page_content attribute

    Returns:
        str: A sanitized string containing all document contents joined with newlines,
             or empty string if formatting fails
    """
    try:
        # Initialize list to store sanitized content
        sanitized_contents = []

        # Process each document
        for doc in docs:
            # Check if document has page_content attribute
            if hasattr(doc, "page_content"):
                # Get content and remove leading/trailing whitespace
                content = doc.page_content.strip()
                # Replace carriage returns and tabs with spaces
                content = content.replace("\r", " ").replace("\t", " ")
                # Add sanitized content to list
                sanitized_contents.append(content)

        # Join all sanitized contents with double newlines
        return "\n\n".join(sanitized_contents)
    except Exception as e:
        # Log any errors and return empty string
        print(f"Error formatting docs: {e}", flush=True)
        return ""


def list_txt_files(data_dir="./data"):
    """
    Recursively find all .txt files in a directory and its subdirectories.

    Args:
        data_dir (str): Path to the directory to search. Defaults to "./data"

    Yields:
        str: String representation of each .txt file path found
    """
    # Get iterator of all .txt files in data_dir and subdirectories
    paths = Path(data_dir).glob("**/*.txt")
    # Yield string version of each path
    for path in paths:
        yield str(path)


def load_csv_files(data_dir="./data"):
    """
    Recursively load all CSV files from a directory and its subdirectories.

    Args:
        data_dir (str): Path to the directory to search. Defaults to "./data"

    Returns:
        list: List of Document objects containing the loaded CSV contents
    """
    # Initialize empty list to store documents
    docs = []

    # Get iterator of all .csv files in data_dir and subdirectories
    paths = Path(data_dir).glob("**/*.csv")

    # Load each CSV file and add to documents list
    for path in paths:
        # Create CSV loader for current file
        loader = CSVLoader(file_path=str(path))
        # Load CSV contents and extend docs list
        docs.extend(loader.load())

    return docs


def get_document_text(uploaded_file, title=None):
    """
    Extract text content from an uploaded file, handling both PDF and text files.

    Args:
        uploaded_file: File object containing the uploaded document
        title (str, optional): Custom title for the document. If not provided,
                             uses the original filename

    Returns:
        list: List of Document objects for PDFs (one per page) or raw text for other files
    """
    # Initialize empty list to store document contents
    docs = []

    # Get filename from uploaded file
    fname = uploaded_file.name

    # Use filename as title if none provided
    if not title:
        title = os.path.basename(fname)

    # Handle PDF files
    if fname.lower().endswith("pdf"):
        # Create PDF reader object
        pdf_reader = PdfReader(uploaded_file)

        # Extract text from each page
        for num, page in enumerate(pdf_reader.pages):
            # Get page text content
            page = page.extract_text()

            # Create Document object with page content and metadata
            doc = Document(page_content=page, metadata={"title": title, "page": (num + 1)})
            docs.append(doc)

    # Handle non-PDF files
    else:
        # Read and decode file content as text
        doc_text = uploaded_file.read().decode()
        docs.append(doc_text)

    return docs

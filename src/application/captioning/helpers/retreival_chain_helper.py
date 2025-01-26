from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages.base import BaseMessage

from application.captioning.models.tokenized_chat_prompt_model import (
    BasicTokenizedChatPromptTemplate,
)
from application.captioning.helpers.document_helper import format_docs


def find_similar(vs, query):
    """
    Find documents similar to the given query using vector store similarity search.

    Args:
        vs: Vector store instance to search in
        query (str): Query text to find similar documents for

    Returns:
        list: List of documents that are semantically similar to the query,
              ordered by similarity score
    """
    docs = vs.similarity_search(query)
    return docs


def get_question(input):
    """
    Extract the question text from various input types.

    Args:
        input: The input to extract the question from. Can be:
            - None: Returns None
            - str: Returns the string directly
            - dict: Returns the value of the 'question' key
            - BaseMessage: Returns the message content

    Returns:
        str or None: The extracted question text, or None if input is None

    Raises:
        Exception: If input is not one of the supported types
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
    # Raise exception for unsupported input types
    else:
        print(f"Input type: {type(input)}", flush=True)
        raise Exception("string or dict with 'question' key expected as RAG chain input.")


def prepare_basic_prompt(tokenizer):
    """
    Prepare a basic prompt template for simple question-answer interactions.

    Returns:
        PromptTemplate: A formatted template that combines:
            - System base message from config
            - Question placeholder
            - Answer prefix for bash scripts
    """
    # Create basic prompt template with provided tokenizer
    basic_prompt = BasicTokenizedChatPromptTemplate(tokenizer)

    # Return the formatted prompt template
    return basic_prompt.get_prompt()

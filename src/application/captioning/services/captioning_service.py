import torch
import os
import json
from typing import Dict, Any
from lightning.pytorch import Trainer

from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory

from application.captioning.helpers.document_helper import save_text_chunks, load_text_chunks, has_dataset_changed, load_txt_files, split_documents
from application.captioning.helpers.vector_db_helper import create_vector_db
from application.captioning.helpers.retreival_chain_helper import (
    prepare_basic_prompt,
)
from application.captioning.helpers.memory_helper import create_memory_chain
from application.captioning.models.captioning_model import CaptioningModule
from application.dataloader.models.dataloader_model import DataloaderModule
from domain.models.captioning.captioning_model import (
    CaptionParameters,
    CaptionDatasetParameters,
)


def load_pipeline():
    try:
        hf = HuggingFacePipeline.from_model_id(
            model_id="bigscience/bloom-1b7",
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 1000},
        )
        return hf
    except Exception as e:
        # Log and return any errors that occur
        print(f"Error loading pipeline: {e}", flush=True)
        return {"error": str(e)}


def load_docs():
    """
    Load documents.
    Currently only supports loading text files, but future support planned for PDF and CSV.

    Returns:
        dict: A dictionary containing either:
            - {"message": "Docs loaded successfully"} on success
            - {"error": <error message>} on failure
    """
    try:
        # Load text files
        docs = load_txt_files()

        # Return success message
        return docs
    except Exception as e:
        # Log error and return error message if loading fails
        print(f"Error loading docs: {e}", flush=True)
        return {"error": str(e)}


def load_ensemble_retriever_from_docs(docs):
    """
    Load an ensemble retriever that combines BM25 and vector similarity search.

    Args:
        ensemble_retriever_parameters: Parameters for configuring the ensemble retriever

    Returns:
        dict: Success/error message
    """

    try:
        # Define paths and configuration
        chunks_path = "./data/store/chunks.pkl"
        collection_name = "chroma"
        # Initialize sentence embeddings model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # Check if we need to reload the dataset
        dataset_changed = has_dataset_changed()
        force_reload = False
        should_reload = force_reload or dataset_changed

        # Handle document chunking based on reload conditions
        if should_reload:
            print("Dataset changes detected or force reload requested", flush=True)
            texts = split_documents(docs)  # Split docs into chunks
            save_text_chunks(texts, chunks_path)  # Cache chunks
        else:
            if os.path.exists(chunks_path):
                print("Loading existing chunks (no dataset changes detected)", flush=True)
                texts = load_text_chunks(chunks_path)
            else:
                print("No existing chunks found, creating new ones", flush=True)
                texts = split_documents(docs)
                save_text_chunks(texts, chunks_path)

        # Create standard vector store
        vs = create_vector_db(texts=texts if should_reload else None, embeddings=embeddings, collection_name=collection_name, force_reload=should_reload)
        # Convert vector store to retriever
        vs_retriever = vs.as_retriever()

        # Create BM25 retriever from document texts
        bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

        # Combine both retrievers with equal weights
        ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vs_retriever], weights=[0.5, 0.5])

        return ensemble_retriever
    except Exception as e:
        print(f"Error loading ensemble retriever: {e}", flush=True)
        return {"error": str(e)}


def load_chain(hf, ensemble_retriever, chain_type: str = "basic"):
    """
    Loads and configures a language model chain based on the specified parameters.

    Args:
        chain_parameters: Configuration parameters for the chain type and behavior

    Returns:
        dict: Status message indicating success or error

    The function supports three types of chains:
    - basic: Simple chain without retrieval
    - rag: Chain with Retrieval Augmented Generation
    - hyde: Chain with Hypothetical Document Embeddings
    """
    try:
        if chain_type == "basic":
            # Create basic chain without retrieval capabilities
            basic_prompt = prepare_basic_prompt()
            chain = create_memory_chain(hf, None, ChatMessageHistory(), basic_prompt)

        else:
            # Configure RAG chain with retrieval capabilities
            rag_prompt = prepare_rag_prompt()
            rag_chain = make_rag_chain(hf, ensemble_retriever, rag_prompt)
            chain = create_memory_chain(hf, rag_chain, ChatMessageHistory())

        return chain
    except Exception as e:
        # Log and return any errors that occur during chain loading
        print(f"Error loading chain: {e}", flush=True)
        return {"error": str(e)}


def caption_midi(parameters: CaptionParameters) -> Dict[str, Any]:
    """
    Generate caption for a single MIDI file.

    Args:
        parameters (CaptionParameters): Configuration parameters including:
            - prompt (str): The prompt to generate caption from
            - model_id (str): HuggingFace model ID
            - max_new_tokens (int): Maximum number of tokens to generate
            - chain_type (str): Type of chain to use (basic or rag)

    Returns:
        Dict[str, Any]: Generated caption or error message
    """
    try:
        # Initialize captioning model
        captioner = CaptioningModule(
            model_id=parameters.model_id,
            max_new_tokens=parameters.max_new_tokens,
            chain_type=parameters.chain_type,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # Generate caption
        result = captioner.caption(parameters.prompt)

        return result
    except Exception as e:
        print(f"Error in caption generation: {str(e)}", flush=True)
        return {"error": True, "message": str(e)}


def caption_dataset(parameters: CaptionDatasetParameters) -> Dict[str, Any]:
    """
    Generate captions for a dataset of MIDI files.

    Args:
        parameters (CaptionDatasetParameters): Configuration parameters including:
            - model_id (str): HuggingFace model ID
            - max_new_tokens (int): Maximum number of tokens to generate
            - chain_type (str): Type of chain to use (basic or rag)
            - force_reload (bool): Whether to force reload document chunks
            - batch_size (int): Number of files to process in parallel

    Returns:
        Dict[str, Any]: Processing summary including results and any errors
    """
    try:
        print("Starting caption_dataset function", flush=True)

        # Load configuration
        with open("shared/assets/config.json", "r") as f:
            config = json.load(f)

        # Get dataloader configuration
        dataloader_config = config["dataloader"]
        print("Loaded configuration", flush=True)

        # Initialize captioning model with configuration
        captioner = CaptioningModule(
            model_id=parameters.model_id,
            max_new_tokens=parameters.max_new_tokens,
            chain_type=parameters.chain_type,
            force_reload=parameters.force_reload,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        print("Initialized captioning model", flush=True)

        # Initialize dataloader module with config parameters
        dataloader = DataloaderModule(
            dataset_name=dataloader_config["dataset_name"],
            context_size=-1,  # Override context_size for captioning
            max_positions=dataloader_config["max_positions"],
            max_bars=dataloader_config["max_bars"],
            max_bars_per_context=dataloader_config["max_bars_per_context"],
            max_contexts_per_file=dataloader_config["max_contexts_per_file"],
            bar_token_mask=dataloader_config["bar_token_mask"],
            bar_token_idx=dataloader_config["bar_token_idx"],
            batch_size=dataloader_config["batch_size"],  # Use batch size from parameters
            num_workers=dataloader_config["num_workers"],
            pin_memory=dataloader_config["pin_memory"],
            train_val_test_split=dataloader_config["train_val_test_split"],
            load_latent=dataloader_config["load_latent"],
            load_symb=dataloader_config["load_symb"],
            load_emotions=dataloader_config["load_emotions"],
            encode=False,
            caption=True,  # Enable caption mode
        )
        print("Initialized dataloader", flush=True)

        # Set up trainer with specific predict configurations
        trainer = Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=1,
            enable_checkpointing=False,
            enable_model_summary=False,
            enable_progress_bar=True,
            logger=False,
            inference_mode=True,  # Ensure we're in inference mode
        )
        print("Initialized trainer", flush=True)

        # Run prediction with explicit dataloader
        print("Starting prediction", flush=True)
        results = trainer.predict(model=captioner, datamodule=dataloader)
        print("Completed prediction", flush=True)

        # Process results
        successful = 0
        errors = []
        outputs = {}

        # Flatten results from all batches
        all_results = [item for batch in results for item in batch] if results else []
        print(f"Processing {len(all_results)} results", flush=True)

        for result in all_results:
            if "error" not in result:
                successful += 1
                outputs[result["file"]] = result["llm_output"]
            else:
                errors.append(f"Error processing {result['file']}: {result['message']}")

        return {
            "Message": "Dataset Captioning Completed",
            "Total": len(all_results),
            "Successful": successful,
            "Failed": len(errors),
            "Errors": errors,
            "Outputs": outputs,
        }

    except Exception as e:
        print(f"Error in dataset captioning: {str(e)}", flush=True)
        return {"error": True, "message": str(e), "details": "An error occurred during dataset captioning"}

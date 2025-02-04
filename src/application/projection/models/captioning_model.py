import os
import torch
import lightning.pytorch as pl
from typing import Dict, Any, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings

from transformers import BloomForCausalLM
from transformers import BloomTokenizerFast

from langchain_huggingface.llms import HuggingFacePipeline
from transformers import BitsAndBytesConfig, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

from application.projection.helpers.document_helper import (
    save_text_chunks,
    load_text_chunks,
    has_dataset_changed,
    load_txt_files,
    split_documents,
)
from application.projection.helpers.vector_db_helper import create_vector_db
from application.projection.helpers.retreival_chain_helper import (
    prepare_basic_prompt,
)


class CaptioningModule(pl.LightningModule):
    def __init__(
        self,
        model_id: str = "bigscience/bloom-1b7",
        max_new_tokens: int = 1000,
        chain_type: str = "basic",
        force_reload: bool = False,
        device: str = "cuda:0",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.chain_type = chain_type
        self.force_reload = force_reload
        self._device = torch.device(device)

        # Initialize components as None
        self.pipeline = None
        self.ensemble_retriever = None
        self.chain = None

        print("CaptioningModule initialized", flush=True)

    def setup(self, stage=None):
        """Initialize all components needed for projection."""
        print(f"Setting up CaptioningModule for stage: {stage}", flush=True)
        if self.pipeline is None:
            self._setup_pipeline()
            print("Done setting up pipeline", flush=True)
        # if self.ensemble_retriever is None:
        #     self._setup_retriever()
        #     print("Done setting up retriever", flush=True)
        if self.chain is None:
            self._setup_chain()
            print("Done setting up chain", flush=True)

    def _setup_pipeline(self):
        """Initialize the HuggingFace pipeline."""
        try:
            # Load tokenizer from local model files
            quantization_config = BitsAndBytesConfig(load_in_4bit=True)

            tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto", quantization_config=quantization_config)

            self.model = model
            self.tokenizer = tokenizer

            # llm = OpenAI(model_name="text-davinci-003")

            # Create HuggingFace pipeline with specified parameters
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.75,  # Reduced from 0.7 to minimize randomness
                top_p=0.9,  # Reduced from 0.9 to be more conservative
                top_k=50,  # Reduced from 50 to limit token choices
                repetition_penalty=1.1,  # Increased from 1.1 to further prevent loops
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                return_full_text=False,  # Only return new tokens
            )

            # Create LangChain pipeline wrapper
            self.pipeline = HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            print(f"Error loading pipeline: {e}", flush=True)
            raise e

    def _setup_retriever(self):
        """Initialize the ensemble retriever."""
        try:
            # Load documents
            docs = load_txt_files()

            # Define paths and configuration
            chunks_path = "./data/store/chunks.pkl"
            collection_name = "chroma"
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2", device=self._device)

            # Check if we need to reload the dataset
            dataset_changed = has_dataset_changed()
            should_reload = self.force_reload or dataset_changed

            # Handle document chunking
            if should_reload:
                print("Dataset changes detected or force reload requested", flush=True)
                texts = split_documents(docs)
                save_text_chunks(texts, chunks_path)
            else:
                if os.path.exists(chunks_path):
                    print("Loading existing chunks", flush=True)
                    texts = load_text_chunks(chunks_path)
                else:
                    print("No existing chunks found, creating new ones", flush=True)
                    texts = split_documents(docs)
                    save_text_chunks(texts, chunks_path)

            # Create vector store
            vs = create_vector_db(texts=texts if should_reload else None, embeddings=embeddings, collection_name=collection_name, force_reload=should_reload)

            # Create BM25 retriever
            bm25_retriever = BM25Retriever.from_texts([t.page_content for t in texts])

            # Create ensemble retriever
            self.ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vs.as_retriever()], weights=[0.5, 0.5])

        except Exception as e:
            print(f"Error loading ensemble retriever: {e}", flush=True)
            raise e

    def _setup_chain(self):
        """Initialize the language model chain."""
        try:
            if self.chain_type == "basic":
                basic_prompt = prepare_basic_prompt(self.tokenizer)
                self.chain = basic_prompt | self.pipeline
        except Exception as e:
            print(f"Error loading chain: {e}", flush=True)
            raise e

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Process a batch for projection using symbolic features and metadata."""
        print(f"Starting predict_step with batch_idx: {batch_idx}", flush=True)
        try:
            # Ensure setup is complete
            self.setup("predict")
            print(f"Setup complete for batch {batch_idx}", flush=True)

            # Process each item in the batch
            results = []
            batch_size = len(batch["files"])
            print(f"Processing {batch_size} files in batch {batch_idx}", flush=True)

            for i in range(batch_size):
                print(f"Processing file {i+1}/{batch_size} in batch {batch_idx}: {batch['files'][i]}", flush=True)

                file = batch["files"][i]

                # Create a dictionary with all features for this item
                item = {
                    "genre": batch["genre"][i] if "genre" in batch else None,
                    "composer": batch["composer"][i] if "composer" in batch else None,
                    "note_density": batch["note_density"][i] if "note_density" in batch else None,
                    "mean_velocity": batch["mean_velocity"][i] if "mean_velocity" in batch else None,
                    "mean_pitch": batch["mean_pitch"][i] if "mean_pitch" in batch else None,
                    "mean_duration": batch["mean_duration"][i] if "mean_duration" in batch else None,
                    "time_signature": batch["time_signatures"][i] if "time_signatures" in batch else None,
                    "key_signature": batch["key_signatures"][i] if "key_signatures" in batch else None,
                    "chords": batch["chords"][i] if "chords" in batch else None,
                    "instruments": batch["instruments"][i] if "instruments" in batch else None,
                    "emotions": batch["emotions"][i] if "emotions" in batch else None,
                }

                # Prepare prompt from features
                print(f"Generated prompt for {file}: {item}", flush=True)

                try:
                    # Generate caption using the chain
                    output = self.chain.invoke(item)
                    print(f"Generated caption for {file}: {output}", flush=True)
                    results.append({"file": file, "llm_output": output})
                except Exception as e:
                    print(f"Error generating caption for {file}: {str(e)}", flush=True)
                    results.append({"file": file, "error": True, "message": str(e)})

            print(f"Completed batch {batch_idx} with {len(results)} results", flush=True)
            return results

        except Exception as e:
            print(f"Error in batch {batch_idx} prediction: {str(e)}", flush=True)
            return [{"error": True, "message": str(e)}]

    def caption(self, prompt: str) -> Dict[str, Any]:
        """Generate caption for a single prompt."""
        try:
            # Ensure everything is set up
            self.setup("predict")

            # Generate response
            output = self.chain.invoke({"question": prompt})
            return {"llm_output": output}
        except Exception as e:
            print(f"Error in caption generation: {str(e)}", flush=True)
            return {"error": True, "message": str(e)}

    def training_step(self, batch, batch_idx):
        # Placeholder for potential future training functionality
        return None

    def validation_step(self, batch, batch_idx):
        # Placeholder for potential future validation functionality
        return None

    def configure_optimizers(self):
        # Placeholder for potential future optimization functionality
        return None

import os
import json
import argparse
from PIL import Image
from tqdm import tqdm
import torch
import clip
import logging
from typing import Dict, Any, Union, List, Optional

from domain.models.pseudo_labeller.pseudo_labeller_model import (
    PseudoLabellerParameters,
    ImageMoodClassificationParameters,
    BatchImageMoodClassificationParameters
)
from domain.constants.classifier_constants import MoodMappingConstants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the mood categories. It is CRITICAL that this list matches the order
# and content expected by the MoodProcessor in your multimodal_mapping module.
MOOD_CATEGORIES = MoodMappingConstants.get_clip_mood_categories()

class ImageMoodClassifier:
    """
    A classifier that uses a pre-trained CLIP model to determine the mood
    of an image based on a predefined set of mood categories.
    
    This class is designed for efficiency by loading the model and pre-computing
    text features only once upon initialization.
    """
    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu", model_name="ViT-B/32", mood_categories=None):
        self.device = device
        self.mood_categories = mood_categories or MOOD_CATEGORIES
        print(f"Using device: {self.device}")

        # Load the CLIP model and the image preprocessor
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        print(f"CLIP model {model_name} loaded successfully.")

        # Create text prompts and tokenize them
        text_prompts = [f"A {mood} photo" for mood in self.mood_categories]
        text_inputs = clip.tokenize(text_prompts).to(self.device)

        # Pre-compute text features for efficiency
        with torch.no_grad():
            self.text_features = self.model.encode_text(text_inputs)
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        print("Text features for mood prompts have been pre-computed.")

    def classify_image(self, image_path: str):
        """
        Processes a single image and returns its mood vector.
        
        Args:
            image_path: The file path to the image.
            
        Returns:
            A list of floats representing the mood vector, or None if an error occurs.
        """
        try:
            # Open and preprocess the image
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)

            # Compute image features and mood probabilities
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
                # Calculate cosine similarity and convert to a probability distribution
                similarity = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
            
            # Convert the mood vector to a standard Python list
            return similarity.squeeze().cpu().numpy().tolist()

        except Exception as e:
            print(f"Warning: Could not process image {image_path}. Error: {e}")
            return None


def create_mood_labels(
    parameters: Union[PseudoLabellerParameters, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create mood labels for a directory of images using BaseModel parameters.
    
    Args:
        parameters: PseudoLabellerParameters or dict with pseudo labeller configuration
        
    Returns:
        Dictionary with labeling status and results
    """
    try:
        # Convert parameters to correct type if needed
        if isinstance(parameters, dict):
            params = PseudoLabellerParameters(**parameters)
        else:
            params = parameters
        
        if not os.path.isdir(params.input_dir):
            raise FileNotFoundError(f"Provided image directory does not exist: {params.input_dir}")

        # Initialize the classifier
        classifier = ImageMoodClassifier(
            device=params.device,
            model_name=params.model_name,
            mood_categories=params.mood_categories
        )

        # Dictionary to store the results
        image_mood_data = {}
        
        # Get a list of all image files in the directory
        image_files = [f for f in os.listdir(params.input_dir) if f.lower().endswith(tuple(params.valid_extensions))]

        if not image_files:
            logger.warning(f"No images found in directory: {params.input_dir}")
            return {
                "status": "warning",
                "message": f"No images found in directory: {params.input_dir}",
                "processed_images": 0,
                "total_images": 0
            }

        # Process each image with a progress bar
        processed_images = 0
        failed_images = []
        
        for filename in tqdm(image_files, desc="Labeling Images"):
            image_path = os.path.join(params.input_dir, filename)
            mood_vector = classifier.classify_image(image_path)
            
            if mood_vector:
                image_mood_data[filename] = mood_vector
                processed_images += 1
            else:
                failed_images.append(filename)

        # Save the final dictionary to a JSON file
        try:
            with open(params.output_file, 'w') as f:
                json.dump(image_mood_data, f, indent=4)
            logger.info(f"Successfully saved mood labels for {processed_images} images to {params.output_file}")
        except Exception as e:
            raise Exception(f"Could not write to output file {params.output_file}. Error: {e}")

        return {
            "status": "success",
            "message": f"Successfully processed {processed_images} images and saved mood labels to {params.output_file}",
            "processed_images": processed_images,
            "failed_images": len(failed_images),
            "total_images": len(image_files),
            "output_file": params.output_file,
            "failed_files": failed_images
        }

    except Exception as e:
        logger.error(f"Failed to create mood labels: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to create mood labels: {str(e)}",
            "error": str(e)
        }


def classify_image_mood(
    parameters: Union[ImageMoodClassificationParameters, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Classify the mood of a single image using BaseModel parameters.
    
    Args:
        parameters: ImageMoodClassificationParameters or dict with classification configuration
        
    Returns:
        Dictionary with classification results
    """
    try:
        # Convert parameters to correct type if needed
        if isinstance(parameters, dict):
            params = ImageMoodClassificationParameters(**parameters)
        else:
            params = parameters
        
        if not os.path.exists(params.image_path):
            raise FileNotFoundError(f"Image file does not exist: {params.image_path}")

        # Initialize the classifier
        classifier = ImageMoodClassifier(
            device=params.device,
            model_name=params.model_name,
            mood_categories=params.mood_categories
        )

        # Classify the image
        mood_vector = classifier.classify_image(params.image_path)
        
        if mood_vector is None:
            raise Exception(f"Failed to classify image: {params.image_path}")

        # Create result dictionary
        result = {
            "status": "success",
            "message": f"Successfully classified image: {params.image_path}",
            "image_path": params.image_path,
            "mood_vector": mood_vector,
            "mood_categories": params.mood_categories
        }

        # Add top predictions if requested
        if params.return_probabilities:
            # Get top k predictions
            import numpy as np
            top_indices = np.argsort(mood_vector)[-params.return_top_k:][::-1]
            top_predictions = [
                {
                    "mood": params.mood_categories[i],
                    "probability": mood_vector[i],
                    "rank": rank + 1
                }
                for rank, i in enumerate(top_indices)
            ]
            result["top_predictions"] = top_predictions

        return result

    except Exception as e:
        logger.error(f"Failed to classify image mood: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to classify image mood: {str(e)}",
            "error": str(e)
        }


def batch_classify_images(
    parameters: Union[BatchImageMoodClassificationParameters, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Classify the mood of multiple images in batch using BaseModel parameters.
    
    Args:
        parameters: BatchImageMoodClassificationParameters or dict with batch classification configuration
        
    Returns:
        Dictionary with batch classification results
    """
    try:
        # Convert parameters to correct type if needed
        if isinstance(parameters, dict):
            params = BatchImageMoodClassificationParameters(**parameters)
        else:
            params = parameters
        
        # Get image files to process
        image_files = []
        
        if params.image_paths:
            # Use specific image paths
            image_files = params.image_paths
        elif params.input_dir:
            # Get all images from directory
            if not os.path.isdir(params.input_dir):
                raise FileNotFoundError(f"Input directory does not exist: {params.input_dir}")
            
            for filename in os.listdir(params.input_dir):
                if filename.lower().endswith(tuple(params.valid_extensions)):
                    image_files.append(os.path.join(params.input_dir, filename))
        else:
            raise ValueError("Either image_paths or input_dir must be provided")

        if not image_files:
            logger.warning("No images found to process")
            return {
                "status": "warning",
                "message": "No images found to process",
                "processed_images": 0,
                "total_images": 0
            }

        # Initialize the classifier
        classifier = ImageMoodClassifier(
            device=params.device,
            model_name=params.model_name,
            mood_categories=params.mood_categories
        )

        # Process images in batches
        processed_images = 0
        failed_images = []
        all_results = {}
        
        for image_path in tqdm(image_files, desc="Processing Images"):
            try:
                mood_vector = classifier.classify_image(image_path)
                
                if mood_vector:
                    image_name = os.path.basename(image_path)
                    result = {
                        "mood_vector": mood_vector,
                        "image_path": image_path
                    }
                    
                    if params.return_probabilities:
                        # Get top predictions
                        import numpy as np
                        top_indices = np.argsort(mood_vector)[-3:][::-1]  # Top 3
                        top_predictions = [
                            {
                                "mood": params.mood_categories[i],
                                "probability": mood_vector[i]
                            }
                            for i in top_indices
                        ]
                        result["top_predictions"] = top_predictions
                    
                    all_results[image_name] = result
                    processed_images += 1
                else:
                    failed_images.append(image_path)
                    
            except Exception as e:
                logger.warning(f"Failed to process image {image_path}: {str(e)}")
                failed_images.append(image_path)

        # Save results if output file is specified
        if params.output_file:
            try:
                with open(params.output_file, 'w') as f:
                    json.dump(all_results, f, indent=4)
                logger.info(f"Saved batch classification results to {params.output_file}")
            except Exception as e:
                logger.warning(f"Failed to save results to {params.output_file}: {str(e)}")

        return {
            "status": "success",
            "message": f"Successfully processed {processed_images} images in batch",
            "processed_images": processed_images,
            "failed_images": len(failed_images),
            "total_images": len(image_files),
            "results": all_results,
            "failed_files": failed_images,
            "output_file": params.output_file
        }

    except Exception as e:
        logger.error(f"Failed to batch classify images: {str(e)}")
        return {
            "status": "error",
            "message": f"Failed to batch classify images: {str(e)}",
            "error": str(e)
        }


def create_mood_labels_for_directory(image_dir: str, output_path: str, device: str):
    """
    Legacy function - processes a directory of images, classifies the mood for each one,
    and saves the results to a JSON file.

    Args:
        image_dir: Path to the directory containing images.
        output_path: Path to save the output JSON file.
        device: The device to run the model on ('cuda' or 'cpu').
    """
    # Convert to new parameter format
    params = PseudoLabellerParameters(
        input_dir=image_dir,
        output_file=output_path,
        device=device
    )
    
    # Use the new function
    result = create_mood_labels(params)
    
    # Print legacy output for backward compatibility
    if result["status"] == "success":
        print(f"\nSuccessfully saved mood labels for {result['processed_images']} images to {output_path}")
    else:
        print(f"\nError: {result['message']}")


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Pseudo-label a directory of images with mood vectors using CLIP.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True, 
        help="Path to the directory containing the images to be labeled."
    )
    
    parser.add_argument(
        "--output_file", 
        type=str, 
        default="image_mood_labels.json", 
        help="Path to the output JSON file where the labels will be saved."
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the CLIP model on (e.g., 'cuda', 'cpu')."
    )
    
    args = parser.parse_args()
    
    create_mood_labels_for_directory(args.input_dir, args.output_file, args.device)


if __name__ == "__main__":
    main() 
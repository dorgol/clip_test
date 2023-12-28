from typing import Tuple

import torch
from PIL import Image
from transformers import AutoProcessor, CLIPVisionModelWithProjection, CLIPProcessor, CLIPModel


def get_model_and_processor(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Loads and returns the CLIP vision model with projection and its associated processor.

    Args:
    model_name (str): The name of the pretrained model to be loaded.

    Returns:
    Tuple[CLIPVisionModelWithProjection, CLIPProcessor]: A tuple containing the loaded model and processor.
    """
    # Load the CLIP vision model with projection using the specified model name
    model = CLIPModel.from_pretrained(model_name)
    # Load the processor for the model using the same model name
    processor = AutoProcessor.from_pretrained(model_name)

    return model, processor


def get_image_embeddings(processor: CLIPProcessor, model: CLIPModel, image: Image.Image) -> \
        torch.Tensor:
    """
    Processes an image and extracts embeddings using the provided model and processor.

    Args:
    processor (CLIPProcessor): The processor associated with the model for image processing.
    model (CLIPVisionModelWithProjection): The CLIP vision model with projection to generate embeddings.
    image (Image.Image): The image for which embeddings are to be generated.

    Returns:
    torch.Tensor: A tensor representing the image embeddings.
    """
    # Process the image using the processor to convert it into a format suitable for the model
    inputs = processor(images=image, return_tensors="pt")
    # Pass the processed image to the model to obtain embeddings
    image_embeds = model.get_image_features(**inputs)

    return image_embeds.detach()
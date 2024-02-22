from typing import Tuple

import torch
import yaml
from transformers import CLIPModel, CLIPConfig
from transformers import CLIPProcessor, AutoTokenizer

# Instantiate model and processor
with open('config.yaml') as f:
    config = yaml.safe_load(f)
MODEL_NAME = config['CLIP_MODEL_STR']


def get_model_and_processor(model_name: str) -> Tuple[CLIPModel, CLIPProcessor]:
    """
    Loads and returns the CLIP text model with projection and its associated processor.

    Args:
        model_name (str): The name of the pretrained model to be loaded.

    Returns:
        Tuple[CLIPTextModelWithProjection, CLIPProcessor]: A tuple containing the loaded model and processor.
    """

    # Load the configuration of the model
    config = CLIPConfig.from_pretrained(model_name)

    # Modify the max_position_embeddings (with potential risks as mentioned)
    config.max_position_embeddings = 1024

    # Load the CLIP text model with the modified configuration
    model = CLIPModel.from_pretrained(model_name, config=config)

    # Load the processor for the model using the same model name
    processor = CLIPProcessor.from_pretrained(model_name)

    return model, processor


def get_text_embeddings(processor: CLIPProcessor, model: CLIPModel, text) -> \
        torch.Tensor:
    """
    Processes an image and extracts embeddings using the provided model and processor.

    Args:
    processor (CLIPProcessor): The processor associated with the model for image processing.
    model (CLIPTextModelWithProjection): The CLIP text model with projection to generate embeddings.
    image (Image.Image): The image for which embeddings are to be generated.

    Returns:
    torch.Tensor: A tensor representing the image embeddings.
    """

    inputs = processor(text, padding=True, return_tensors="pt")
    # Pass the processed image to the model to obtain embeddings
    text_embeds = model.get_text_features(**inputs)

    return text_embeds.detach()

from diffusers import StableDiffusionPipeline
import torch
import base64
import json
import logging
import numpy as np

def process_data(data: dict) -> dict:
    return {
        "prompt": [data.pop("prompt", data)] * min(data.pop("number", 2), 5),
        "guidance_scale": data.pop("guidance_scale", 7.5),
        "num_inference_steps": min(data.pop("num_inference_steps", 50), 50),
        "height": 512,
        "width": 512,
    }

def model_fn(model_dir: str):
    logging.info(f"Loading model from {model_dir=}")
    t2i_pipe:StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        model_dir,
    )
    if torch.cuda.is_available():
        t2i_pipe = t2i_pipe.to("cuda")

    t2i_pipe.enable_attention_slicing()
    logging.info(f"Model was loaded from {model_dir=}")
    return t2i_pipe

def input_fn(input_data, content_type):
    logging.debug(f"Run input_fn. {input_data=}, {content_type=}")
    assert content_type == 'application/json', "Unexpected {content_type=}. Need application/json"
    data = json.loads(input_data)
    return data

def predict_fn(data: dict, hgf_pipe) -> dict:
    with torch.autocast("cuda"):
        images = hgf_pipe(**process_data(data))["images"]

    # return dictionary, which will be json serializable
    return {
        "images": [
            base64.b64encode(np.array(image).astype(np.uint8)).decode("utf-8")
            for image in images
        ]
    }

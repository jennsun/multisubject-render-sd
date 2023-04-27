import gradio as gr
import torch
import argparse
from omegaconf import OmegaConf
from gligen.task_grounded_generation import grounded_generation_box, load_ckpt
from ldm.util import default_device

import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from functools import partial
import math
from contextlib import nullcontext

from gradio import processing_utils
from typing import Optional

from huggingface_hub import hf_hub_download
hf_hub_download = partial(hf_hub_download, library_name="gligen_demo")

import openai
from gradio.components import Textbox, Text
# import os

arg_bool = lambda x: x.lower() == 'true'
device = default_device()

print(f"GLIGEN uses {device.upper()} device.")
if device == "cpu":
    print("It will be sloooow. Consider using GPU support with CUDA or (in case of M1/M2 Apple Silicon) MPS.")
elif device == "mps":
    print("The fastest you can get on M1/2 Apple Silicon. Yet, still many opimizations are switched off and it will is much slower than CUDA.")

def parse_option():
    parser = argparse.ArgumentParser('GLIGen Demo', add_help=False)
    parser.add_argument("--folder", type=str,  default="create_samples", help="path to OUTPUT")
    parser.add_argument("--official_ckpt", type=str,  default='ckpts/sd-v1-4.ckpt', help="")
    parser.add_argument("--guidance_scale", type=float,  default=5, help="")
    parser.add_argument("--alpha_scale", type=float,  default=1, help="scale tanh(alpha). If 0, the behaviour is same as original model")
    parser.add_argument("--load-text-box-generation", type=arg_bool, default=True, help="Load text-box generation pipeline.")
    parser.add_argument("--load-text-box-inpainting", type=arg_bool, default=False, help="Load text-box inpainting pipeline.")
    parser.add_argument("--load-text-image-box-generation", type=arg_bool, default=False, help="Load text-image-box generation pipeline.")
    args = parser.parse_args()
    return args
args = parse_option()

def load_from_hf(repo_id, filename='diffusion_pytorch_model.bin'):
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    return torch.load(cache_file, map_location='cpu')

def load_ckpt_config_from_hf(modality):
    ckpt = load_from_hf(f'gligen/{modality}')
    config = load_from_hf('gligen/demo_config_legacy', filename=f'{modality}.pth')
    return ckpt, config

if args.load_text_box_generation:
    pretrained_ckpt_gligen, config = load_ckpt_config_from_hf('gligen-generation-text-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = False
    config.model['params']['is_style'] = False
    loaded_model_list = load_ckpt(config, pretrained_ckpt_gligen) 


if args.load_text_box_inpainting:
    pretrained_ckpt_gligen_inpaint, config = load_ckpt_config_from_hf('gligen-inpainting-text-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = True 
    config.model['params']['is_style'] = False
    loaded_model_list_inpaint = load_ckpt(config, pretrained_ckpt_gligen_inpaint)


if args.load_text_image_box_generation:
    pretrained_ckpt_gligen_style, config = load_ckpt_config_from_hf('gligen-generation-text-image-box')
    config = OmegaConf.create( config["_content"] ) # config used in training
    config.update( vars(args) )
    config.model['params']['is_inpaint'] = False 
    config.model['params']['is_style'] = True
    loaded_model_list_style = load_ckpt(config, pretrained_ckpt_gligen_style)


def load_clip_model():
    from transformers import CLIPProcessor, CLIPModel
    version = "openai/clip-vit-large-patch14"
    model = CLIPModel.from_pretrained(version).to(device)
    processor = CLIPProcessor.from_pretrained(version)

    return {
        'version': version,
        'model': model,
        'processor': processor,
    }

clip_model = load_clip_model()

class ImageMask(gr.components.Image):
    """
    Sets: source="canvas", tool="sketch"
    """

    is_template = True

    def __init__(self, **kwargs):
        super().__init__(source="upload", tool="sketch", interactive=True, **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        if self.tool == "sketch" and self.source in ["upload", "webcam"] and type(x) != dict:
            decode_image = processing_utils.decode_base64_to_image(x)
            width, height = decode_image.size
            mask = np.zeros((height, width, 4), dtype=np.uint8)
            mask[..., -1] = 255
            mask = self.postprocess(mask)
            x = {'image': x, 'mask': mask}
        return super().preprocess(x)


################

title = "Multiple Interfaces"

#app 1
openai.api_key = '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'
prompt_base = 'Separate the subjects in this sentence by semicolons. Include action verbs that correspond to each subject if necessary. For example, the sentence "a tiger and a horse running in a greenland" should output "tiger; horse". If there are numbers, make each subject unique. For example, "2 dogs and 1 duck" would be "dog; dog; duck." If the sentence is "a cowboy and a ninja fighting" output "cowboy fighting; ninja fighting." Do the same for the following sentence: \n'
def separate_subjects(input_text):
    prompt = prompt_base + input_text
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    output_text = response.choices[0].text.strip()
    return output_text

#interface 1
app1 =  gr.Interface(fn = separate_subjects, inputs="text", outputs="text")
#interface 2

app2 =  gr.Interface(fn = user_help, inputs="text", outputs="text")

demo = gr.TabbedInterface([app1, app2], ["V1", "What to do"])

demo.launch()
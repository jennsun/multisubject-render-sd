import gradio as gr	
import torch	
from omegaconf import OmegaConf	
from gligen.task_grounded_generation import grounded_generation_box, load_ckpt, load_common_ckpt	
import json	
import numpy as np	
from PIL import Image, ImageDraw, ImageFont	
from functools import partial	
from collections import Counter	
import math	
import gc	
from gradio import processing_utils	
from typing import Optional	
import warnings	
from datetime import datetime	
from huggingface_hub import hf_hub_download	
hf_hub_download = partial(hf_hub_download, library_name="gligen_demo")	
import sys	

import os
import openai
from gradio.components import Textbox, Text

sys.tracebacklimit = 0	

def load_from_hf(repo_id, filename='diffusion_pytorch_model.bin', subfolder=None):	
    cache_file = hf_hub_download(repo_id=repo_id, filename=filename, subfolder=subfolder)	
    return torch.load(cache_file, map_location='cpu')	
def load_ckpt_config_from_hf(modality):	
    ckpt = load_from_hf('gligen/demo_ckpts_legacy', filename=f'{modality}.pth', subfolder='model')	
    config = load_from_hf('gligen/demo_ckpts_legacy', filename=f'{modality}.pth', subfolder='config')	
    return ckpt, config	
def ckpt_load_helper(modality, is_inpaint, is_style, common_instances=None):	
    pretrained_ckpt_gligen, config = load_ckpt_config_from_hf(modality)	
    config = OmegaConf.create( config["_content"] ) # config used in training	
    config.alpha_scale = 1.0	
    config.model['params']['is_inpaint'] = is_inpaint	
    config.model['params']['is_style'] = is_style	
    if common_instances is None:	
        common_ckpt = load_from_hf('gligen/demo_ckpts_legacy', filename=f'common.pth', subfolder='model')	
        common_instances = load_common_ckpt(config, common_ckpt)	
    loaded_model_list = load_ckpt(config, pretrained_ckpt_gligen, common_instances)	
    return loaded_model_list, common_instances	
class Instance:	
    def __init__(self, capacity = 2):	
        self.model_type = 'base'	
        self.loaded_model_list = {}	
        self.counter = Counter()	
        self.global_counter = Counter()	
        self.loaded_model_list['base'], self.common_instances = ckpt_load_helper(	
            'gligen-generation-text-box',	
            is_inpaint=False, is_style=False, common_instances=None	
        )	
        self.capacity = capacity	
    def _log(self, model_type, batch_size, instruction, phrase_list):	
        self.counter[model_type] += 1	
        self.global_counter[model_type] += 1	
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")	
        print('[{}] Current: {}, All: {}. Samples: {}, prompt: {}, phrases: {}'.format(	
            current_time, dict(self.counter), dict(self.global_counter), batch_size, instruction, phrase_list	
        ))	
    def get_model(self, model_type, batch_size, instruction, phrase_list):	
        if model_type in self.loaded_model_list:	
            self._log(model_type, batch_size, instruction, phrase_list)	
            return self.loaded_model_list[model_type]	
        if self.capacity == len(self.loaded_model_list):	
            least_used_type = self.counter.most_common()[-1][0]	
            del self.loaded_model_list[least_used_type]	
            del self.counter[least_used_type]	
            gc.collect()	
            torch.cuda.empty_cache()	
        self.loaded_model_list[model_type] = self._get_model(model_type)	
        self._log(model_type, batch_size, instruction, phrase_list)	
        return self.loaded_model_list[model_type]	
    def _get_model(self, model_type):	
        if model_type == 'base':	
            return ckpt_load_helper(	
                'gligen-generation-text-box',	
                is_inpaint=False, is_style=False, common_instances=self.common_instances	
            )[0]	
        elif model_type == 'inpaint':	
            return ckpt_load_helper(	
                'gligen-inpainting-text-box',	
                is_inpaint=True, is_style=False, common_instances=self.common_instances	
            )[0]	
        elif model_type == 'style':	
            return ckpt_load_helper(	
                'gligen-generation-text-image-box',	
                is_inpaint=False, is_style=True, common_instances=self.common_instances	
            )[0]	
        	
        assert False	
instance = Instance()	
def load_clip_model():	
    from transformers import CLIPProcessor, CLIPModel	
    version = "openai/clip-vit-large-patch14"	
    model = CLIPModel.from_pretrained(version).cuda()	
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


class Blocks(gr.Blocks):

    def __init__(
        self,
        theme: str = "default",
        analytics_enabled: Optional[bool] = None,
        mode: str = "blocks",
        title: str = "Gradio",
        css: Optional[str] = None,
        **kwargs,
    ):

        self.extra_configs = {
            'thumbnail': kwargs.pop('thumbnail', ''),
            'url': kwargs.pop('url', 'https://gradio.app/'),
            'creator': kwargs.pop('creator', 'Jenny Sun'),
        }

        super(Blocks, self).__init__(theme, analytics_enabled, mode, title, css, **kwargs)
        warnings.filterwarnings("ignore")
    def get_config_file(self):
        config = super(Blocks, self).get_config_file()

        for k, v in self.extra_configs.items():
            config[k] = v
        
        return config

'''
inference model
'''

@torch.no_grad()
def inference(task, language_instruction, grounding_instruction, inpainting_boxes_nodrop, image,
              alpha_sample, guidance_scale, batch_size,
              fix_seed, rand_seed, actual_mask, style_image,
              *args, **kwargs):
    grounding_instruction = json.loads(grounding_instruction)
    phrase_list, location_list = [], []
    for k, v  in grounding_instruction.items():
        phrase_list.append(k)
        location_list.append(v)

    placeholder_image = Image.open('images/teddy.jpg').convert("RGB")    
    image_list = [placeholder_image] * len(phrase_list) # placeholder input for visual prompt, which is disabled

    batch_size = int(batch_size)
    if not 1 <= batch_size <= 4:
        batch_size = 2

    if style_image == None:
        has_text_mask = 1 
        has_image_mask = 0 # then we hack above 'image_list' 
    else:
        valid_phrase_len = len(phrase_list)

        phrase_list += ['placeholder']
        has_text_mask = [1]*valid_phrase_len + [0]

        image_list = [placeholder_image]*valid_phrase_len + [style_image]
        has_image_mask = [0]*valid_phrase_len + [1]
        
        location_list += [ [0.0, 0.0, 1, 0.01]  ] # style image grounding location

    if task == 'Grounded Inpainting':
        alpha_sample = 1.0

    instruction = dict(
        prompt = language_instruction,
        phrases = phrase_list,
        images = image_list,
        locations = location_list,
        alpha_type = [alpha_sample, 0, 1.0 - alpha_sample], 
        has_text_mask = has_text_mask,
        has_image_mask = has_image_mask,
        save_folder_name = language_instruction,
        guidance_scale = guidance_scale,
        batch_size = batch_size,
        fix_seed = bool(fix_seed),
        rand_seed = int(rand_seed),
        actual_mask = actual_mask,
        inpainting_boxes_nodrop = inpainting_boxes_nodrop,
    )

    print("instruction values", instruction)

    get_model = partial(instance.get_model,	
                            batch_size=batch_size,	
                            instruction=language_instruction,	
                            phrase_list=phrase_list)	
    with torch.autocast(device_type='cuda', dtype=torch.float16):	
        if task == 'Grounded Generation':	
            if style_image == None:	
                return grounded_generation_box(get_model('base'), instruction, *args, **kwargs)	
            else:	
                return grounded_generation_box(get_model('style'), instruction, *args, **kwargs)	
        elif task == 'Grounded Inpainting':	
            assert image is not None	
            instruction['input_image'] = image.convert("RGB")	
            return grounded_generation_box(get_model('inpaint'), instruction, *args, **kwargs)


def draw_box(boxes=[], texts=[], img=None):
    if len(boxes) == 0 and img is None:
        return None

    if img is None:
        img = Image.new('RGB', (512, 512), (255, 255, 255))
    colors = ["red", "olive", "blue", "green", "orange", "brown", "cyan", "purple"]
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("DejaVuSansMono.ttf", size=18)
    for bid, box in enumerate(boxes):
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=colors[bid % len(colors)], width=4)
        anno_text = texts[bid]
        draw.rectangle([box[0], box[3] - int(font.size * 1.2), box[0] + int((len(anno_text) + 0.8) * font.size * 0.6), box[3]], outline=colors[bid % len(colors)], fill=colors[bid % len(colors)], width=4)
        draw.text([box[0] + int(font.size * 0.2), box[3] - int(font.size*1.2)], anno_text, font=font, fill=(255,255,255))
    return img

def get_concat(ims):
    if len(ims) == 1:
        n_col = 1
    else:
        n_col = 2
    n_row = math.ceil(len(ims) / 2)
    dst = Image.new('RGB', (ims[0].width * n_col, ims[0].height * n_row), color="white")
    for i, im in enumerate(ims):
        row_id = i // n_col
        col_id = i % n_col
        dst.paste(im, (im.width * col_id, im.height * row_id))
    return dst


def auto_append_grounding(language_instruction, grounding_texts):
    for grounding_text in grounding_texts:
        if grounding_text not in language_instruction and grounding_text != 'auto':
            language_instruction += "; " + grounding_text
    print(language_instruction)
    return language_instruction

def generate(task, language_instruction, grounding_texts, sketch_pad,
             alpha_sample, guidance_scale, batch_size,
             fix_seed, rand_seed, use_actual_mask, append_grounding, style_cond_image,
             state):
    if 'boxes' not in state:
        state['boxes'] = []

    boxes = state['boxes']
    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    assert len(boxes) == len(grounding_texts)
    if len(boxes) != len(grounding_texts):	
        if len(boxes) < len(grounding_texts):	
            raise ValueError("""The number of boxes should be equal to the number of grounding objects.	
Number of boxes drawn: {}, number of grounding tokens: {}.	
Please draw boxes accordingly on the sketch pad.""".format(len(boxes), len(grounding_texts)))	
        grounding_texts = grounding_texts + [""] * (len(boxes) - len(grounding_texts))


    boxes = (np.asarray(boxes) / 512).tolist()
    grounding_instruction = json.dumps({obj: box for obj,box in zip(grounding_texts, boxes)})
    print("GROUNDING instruction -- should be separated text semicolon", grounding_instruction)

    image = None
    actual_mask = None
    if task == 'Grounded Inpainting':
        image = state.get('original_image', sketch_pad['image']).copy()
        image = center_crop(image)
        image = Image.fromarray(image)

        if use_actual_mask:
            actual_mask = sketch_pad['mask'].copy()
            if actual_mask.ndim == 3:
                actual_mask = actual_mask[..., 0]
            actual_mask = center_crop(actual_mask, tgt_size=(64, 64))
            actual_mask = torch.from_numpy(actual_mask == 0).float()

        if state.get('inpaint_hw', None):
            boxes = np.asarray(boxes) * 0.9 + 0.05
            boxes = boxes.tolist()
            grounding_instruction = json.dumps({obj: box for obj,box in zip(grounding_texts, boxes) if obj != 'auto'})
    
    # Try to remove append grounding
    # if append_grounding:
    #     language_instruction = auto_append_grounding(language_instruction, grounding_texts)

    gen_images, gen_overlays = inference(
        task, language_instruction, grounding_instruction, boxes, image,
        alpha_sample, guidance_scale, batch_size,
        fix_seed, rand_seed, actual_mask, style_cond_image, clip_model=clip_model,
    )

    for idx, gen_image in enumerate(gen_images):

        if task == 'Grounded Inpainting' and state.get('inpaint_hw', None):
            hw = min(*state['original_image'].shape[:2])
            gen_image = sized_center_fill(state['original_image'].copy(), np.array(gen_image.resize((hw, hw))), hw, hw)
            gen_image = Image.fromarray(gen_image)
        
        gen_images[idx] = gen_image

    blank_samples = batch_size % 2 if batch_size > 1 else 0
    gen_images = [gr.Image.update(value=x, visible=True) for i,x in enumerate(gen_images)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(blank_samples)] \
                    + [gr.Image.update(value=None, visible=False) for _ in range(4 - batch_size - blank_samples)]

    return gen_images + [state]


def binarize(x):
    return (x != 0).astype('uint8') * 255

def sized_center_crop(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    return img[starty:starty+cropy, startx:startx+cropx]

def sized_center_fill(img, fill, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    img[starty:starty+cropy, startx:startx+cropx] = fill
    return img

def sized_center_mask(img, cropx, cropy):
    y, x = img.shape[:2]
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)    
    center_region = img[starty:starty+cropy, startx:startx+cropx].copy()
    img = (img * 0.2).astype('uint8')
    img[starty:starty+cropy, startx:startx+cropx] = center_region
    return img

def center_crop(img, HW=None, tgt_size=(512, 512)):
    if HW is None:
        H, W = img.shape[:2]
        HW = min(H, W)
    img = sized_center_crop(img, HW, HW)
    img = Image.fromarray(img)
    img = img.resize(tgt_size)
    return np.array(img)

def draw(task, input, grounding_texts, new_image_trigger, state):
    if type(input) == dict:
        image = input['image']
        mask = input['mask']
    else:
        mask = input

    if mask.ndim == 3:
        mask = mask[..., 0]

    image_scale = 1.0

    # resize trigger
    if task == "Grounded Inpainting":
        mask_cond = mask.sum() == 0
        # size_cond = mask.shape != (512, 512)
        if mask_cond and 'original_image' not in state:
            image = Image.fromarray(image)
            width, height = image.size
            scale = 600 / min(width, height)
            image = image.resize((int(width * scale), int(height * scale)))
            state['original_image'] = np.array(image).copy()
            image_scale = float(height / width)
            return [None, new_image_trigger + 1, image_scale, state]
        else:
            original_image = state['original_image']
            H, W = original_image.shape[:2]
            image_scale = float(H / W)

    mask = binarize(mask)
    if mask.shape != (512, 512):
        # assert False, "should not receive any non- 512x512 masks."
        if 'original_image' in state and state['original_image'].shape[:2] == mask.shape:
            mask = center_crop(mask, state['inpaint_hw'])
            image = center_crop(state['original_image'], state['inpaint_hw'])
        else:
            mask = np.zeros((512, 512), dtype=np.uint8)
    # mask = center_crop(mask)
    mask = binarize(mask)

    if type(mask) != np.ndarray:
        mask = np.array(mask)

    if mask.sum() == 0 and task != "Grounded Inpainting":
        state = {}

    if task != 'Grounded Inpainting':
        image = None
    else:
        image = Image.fromarray(image)

    if 'boxes' not in state:
        state['boxes'] = []

    if 'masks' not in state or len(state['masks']) == 0:
        state['masks'] = []
        last_mask = np.zeros_like(mask)
    else:
        last_mask = state['masks'][-1]

    if type(mask) == np.ndarray and mask.size > 1:
        diff_mask = mask - last_mask
    else:
        diff_mask = np.zeros([])

    if diff_mask.sum() > 0:
        x1x2 = np.where(diff_mask.max(0) != 0)[0]
        y1y2 = np.where(diff_mask.max(1) != 0)[0]
        y1, y2 = y1y2.min(), y1y2.max()
        x1, x2 = x1x2.min(), x1x2.max()

        if (x2 - x1 > 5) and (y2 - y1 > 5):
            state['masks'].append(mask.copy())
            state['boxes'].append((x1, y1, x2, y2))

    grounding_texts = [x.strip() for x in grounding_texts.split(';')]
    grounding_texts = [x for x in grounding_texts if len(x) > 0]
    if len(grounding_texts) < len(state['boxes']):
        grounding_texts += [f'Obj. {bid+1}' for bid in range(len(grounding_texts), len(state['boxes']))]

    box_image = draw_box(state['boxes'], grounding_texts, image)

    if box_image is not None and state.get('inpaint_hw', None):
        inpaint_hw = state['inpaint_hw']
        box_image_resize = np.array(box_image.resize((inpaint_hw, inpaint_hw)))
        original_image = state['original_image'].copy()
        box_image = sized_center_fill(original_image, box_image_resize, inpaint_hw, inpaint_hw)

    return [box_image, new_image_trigger, image_scale, state]

def clear(task, sketch_pad_trigger, batch_size, state, switch_task=False):
    if task != 'Grounded Inpainting':
        sketch_pad_trigger = sketch_pad_trigger + 1
    blank_samples = batch_size % 2 if batch_size > 1 else 0
    out_images = [gr.Image.update(value=None, visible=True) for i in range(batch_size)] \
                    + [gr.Image.update(value=None, visible=True) for _ in range(blank_samples)] \
                    + [gr.Image.update(value=None, visible=False) for _ in range(4 - batch_size - blank_samples)]
    state = {}
    return [None, sketch_pad_trigger, None, 1.0] + out_images + [state]

css = """	
#img2img_image, #img2img_image > .fixed-height, #img2img_image > .fixed-height > div, #img2img_image > .fixed-height > div > img	
{	
    height: var(--height) !important;	
    max-height: var(--height) !important;	
    min-height: var(--height) !important;	
}	
#paper-info a {	
    color:#008AD7;	
    text-decoration: none;	
}	
#paper-info a:hover {	
    cursor: pointer;	
    text-decoration: none;	
}	
"""

rescale_js = """
function(x) {
    const root = document.querySelector('gradio-app').shadowRoot || document.querySelector('gradio-app');
    let image_scale = parseFloat(root.querySelector('#image_scale input').value) || 1.0;
    const image_width = root.querySelector('#img2img_image').clientWidth;
    const target_height = parseInt(image_width * image_scale);
    document.body.style.setProperty('--height', `${target_height}px`);
    root.querySelectorAll('button.justify-center.rounded')[0].style.display='none';
    root.querySelectorAll('button.justify-center.rounded')[1].style.display='none';
    return x;
}
"""

# Set up OpenAI API key
openai.api_key = os.environ['OPENAI_API_KEY']

prompt_base = 'Separate the subjects in this sentence by semicolons. For example, the sentence "a tiger and a horse running in a greenland" should output "tiger; horse". If there are numbers, make each subject unique. For example, "2 dogs and 1 duck" would be "dog; dog; duck." Do the same for the following sentence: \n'

original_input = ""
separated_subjects = ""

# language_instruction = gr.Textbox(
#     label="Language Instruction by User",
#     value="2 horses running",
#     visible=False
# )
# grounding_instruction = gr.Textbox(
#     label="Subjects in image (Separated by semicolon)",
#     value="horse; horse",
#     visible=False
# )

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

with Blocks(
    css=css,
    analytics_enabled=False,
    title="GLIGen demo",
) as main:
    gr.Markdown('<h1 style="text-align: center;">MSR: MultiSubject Render</h1>')
    gr.Markdown('<h3 style="text-align: center;">Using NLP and Grounding Processing Techniques to improve image generation of multiple subjects with base Stable Diffusion Model</h3>')
    
    with gr.Row():
        with gr.Column(scale=4):
            sketch_pad_trigger = gr.Number(value=0, visible=False)
            sketch_pad_resize_trigger = gr.Number(value=0, visible=False)
            init_white_trigger = gr.Number(value=0, visible=False)
            image_scale = gr.Number(value=0, elem_id="image_scale", visible=False)
            new_image_trigger = gr.Number(value=0, visible=False)

            # UNCOMMENT THIS WHEN YOU WANT TO TOGGLE INPAINTING OPTION
            task = gr.Radio(
                choices=["Version 1: Single Layer", 'Version 2: Inpainting w/ Multiple Layers'],
                type="value",
                value="Grounded Generation",
                label="Task",
                visible=False,
            )

            # language_instruction = gr.Textbox(
            #     label="Enter your prompt here",
            # )
            # grounding_instruction = gr.Textbox(
            #     label="Grounding instruction (Separated by semicolon)",
            # )
            # grounding_instruction = separate_subjects(language_instruction.value)
            # print(f"The user entered: {language_instruction}")
            # print(f"Our function gave: {grounding_instruction}")

            # EXPERIMENTING:
            with gr.Column():
                user_input = gr.Text(label="Enter your prompt here:")
            gr.Examples(["2 horses running", "A cowboy and ninja fighting", "An apple and an orange on a table"], inputs=[user_input])
            with gr.Column():
                btn = gr.Button("Gen")
            with gr.Column():
                separated_text = gr.Text(label="Subjects Separated by Semicolon")
            btn.click(separate_subjects, inputs=[user_input], outputs=[separated_text])
            # language_instruction = gr.Textbox(
            #     label="Language Instruction by User",
            #     value=seed,
            #     visible=False
            # )
            print("separated_text", separated_text)
            language_instruction=user_input
            grounding_instruction=separated_text
            print("language_instruction after blocks: ", language_instruction)
            print("grounding_instruction after blocks: ", language_instruction)
            # language_instruction.value = seed
            # grounding_instruction.value = separated_text
            
            ####################
            # language_instruction = gr.Textbox(
            #     label="Enter your prompt here",
            # )
            # original_input = language_instruction.value
            # start_btn = gr.Button('Start')
            # start_btn.click(update_grounding_instruction)
            # print("separated subjects 2:", separated_subjects)

            # language_instruction = gr.Textbox(
            #     label="just needs to be here",
            #     value=seed,
            #     visible=False
            # )
            # grounding_instruction = gr.Textbox(
            #     label="Subjects in image (Separated by semicolon)",
            #     value=separated_text,
            #     visible=False
            # )
            
            # print("Language instruction Value:", language_instruction.value)
            # print("Grounding instruction:", grounding_instruction.value)


            ####################

            with gr.Row():
                sketch_pad = ImageMask(label="Sketch Pad", elem_id="img2img_image")
                out_imagebox = gr.Image(type="pil", label="Parsed Sketch Pad")
            with gr.Row():
                clear_btn = gr.Button(value='Clear')
                gen_btn = gr.Button(value='Generate')
            with gr.Accordion("Advanced Options", open=False):
                with gr.Column():
                    alpha_sample = gr.Slider(minimum=0, maximum=1.0, step=0.1, value=0.3, label="Scheduled Sampling (τ)", visible=False)
                    guidance_scale = gr.Slider(minimum=0, maximum=50, step=0.5, value=20, label="Guidance Scale (how closely it adheres to your prompt)")
                    batch_size = gr.Slider(minimum=1, maximum=4, step=1, value=4, label="Number of Images")
                    append_grounding = gr.Checkbox(value=True, label="Append grounding instructions to the caption", visible=False)
                    use_actual_mask = gr.Checkbox(value=False, label="Use actual mask for inpainting", visible=False)
                    with gr.Row():
                        fix_seed = gr.Checkbox(value=False, label="Fixed seed", visible=False)
                        rand_seed = gr.Slider(minimum=0, maximum=1000, step=1, value=0, label="Seed", visible=False)
                    with gr.Row():
                        use_style_cond = gr.Checkbox(value=False, label="Enable Style Condition", visible=False)
                        style_cond_image = gr.Image(type="pil", label="Style Condition", interactive=True, visible=False)
        with gr.Column(scale=4):
            gr.HTML('<span style="font-size: 20px; font-weight: bold">Generated Images</span>')
            with gr.Row():
                out_gen_1 = gr.Image(type="pil", visible=True, show_label=False)
                out_gen_2 = gr.Image(type="pil", visible=True, show_label=False)
            with gr.Row():
                out_gen_3 = gr.Image(type="pil", visible=True, show_label=False)
                out_gen_4 = gr.Image(type="pil", visible=True, show_label=False)

        state = gr.State({})

        class Controller:
            def __init__(self):
                self.calls = 0
                self.tracks = 0
                self.resizes = 0
                self.scales = 0

            def init_white(self, init_white_trigger):
                self.calls += 1
                return np.ones((512, 512), dtype='uint8') * 255, 1.0, init_white_trigger+1

            def change_n_samples(self, n_samples):
                blank_samples = n_samples % 2 if n_samples > 1 else 0
                return [gr.Image.update(visible=True) for _ in range(n_samples + blank_samples)] \
                    + [gr.Image.update(visible=False) for _ in range(4 - n_samples - blank_samples)]

            def resize_centercrop(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_cc = center_crop(image, inpaint_hw)
                # print(f'resize triggered {self.resizes}', image.shape, '->', image_cc.shape)
                return image_cc, state

            def resize_masked(self, state):
                self.resizes += 1
                image = state['original_image'].copy()
                inpaint_hw = int(0.9 * min(*image.shape[:2]))
                state['inpaint_hw'] = inpaint_hw
                image_mask = sized_center_mask(image, inpaint_hw, inpaint_hw)
                state['masked_image'] = image_mask.copy()
                # print(f'mask triggered {self.resizes}')
                return image_mask, state
            
            def switch_task_hide_cond(self, task):
                cond = False
                if task == "Grounded Generation":
                    cond = True

                return gr.Checkbox.update(visible=cond, value=False), gr.Image.update(value=None, visible=False), gr.Slider.update(visible=cond), gr.Checkbox.update(visible=(not cond), value=False)

        controller = Controller()
        main.load(
            lambda x:x+1,
            inputs=sketch_pad_trigger,
            outputs=sketch_pad_trigger,
            queue=False)
        sketch_pad.edit(
            draw,
            inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        grounding_instruction.change(
            draw,
            inputs=[task, sketch_pad, grounding_instruction, sketch_pad_resize_trigger, state],
            outputs=[out_imagebox, sketch_pad_resize_trigger, image_scale, state],
            queue=False,
        )
        clear_btn.click(
            clear,
            inputs=[task, sketch_pad_trigger, batch_size, state],
            outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
            queue=False)
        task.change(
            partial(clear, switch_task=True),
            inputs=[task, sketch_pad_trigger, batch_size, state],
            outputs=[sketch_pad, sketch_pad_trigger, out_imagebox, image_scale, out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
            queue=False)
        sketch_pad_trigger.change(
            controller.init_white,
            inputs=[init_white_trigger],
            outputs=[sketch_pad, image_scale, init_white_trigger],
            queue=False)
        sketch_pad_resize_trigger.change(
            controller.resize_masked,
            inputs=[state],
            outputs=[sketch_pad, state],
            queue=False)
        batch_size.change(
            controller.change_n_samples,
            inputs=[batch_size],
            outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4],
            queue=False)
        gen_btn.click(
            generate,
            inputs=[
                task, language_instruction, grounding_instruction, sketch_pad,
                alpha_sample, guidance_scale, batch_size,
                fix_seed, rand_seed,
                use_actual_mask,
                append_grounding, style_cond_image,
                state,
            ],
            outputs=[out_gen_1, out_gen_2, out_gen_3, out_gen_4, state],
            queue=True
        )
        # start_btn.click(
        #     update_grounding_instruction,
        #     # inputs=[
        #     #     original_input,
        #     # ],
        #     # outputs=[separated_subjects],
        #     # queue=True
        # )
        sketch_pad_resize_trigger.change(
            None,
            None,
            sketch_pad_resize_trigger,
            _js=rescale_js,
            queue=False)
        init_white_trigger.change(
            None,
            None,
            init_white_trigger,
            _js=rescale_js,
            queue=False)
        use_style_cond.change(
            lambda cond: gr.Image.update(visible=cond),
            use_style_cond,
            style_cond_image,
            queue=False)
        task.change(
            controller.switch_task_hide_cond,
            inputs=task,
            outputs=[use_style_cond, style_cond_image, alpha_sample, use_actual_mask],
            queue=False)

main.queue(concurrency_count=1, api_open=False)
main.launch(share=False, show_api=False, show_error=True)
import torch
from PIL import Image
import matplotlib.pyplot as plt
from common.processors import BlipCaptionProcessor, BlipDiffusionInputImageProcessor
from models.blip_diffusion_models.blip_diffusion import BlipDiffusion
from common.utils import load_cfg


cfg = load_cfg('./config/inference_settings.yaml')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

weight_dtype = torch.float32
if cfg['model']['mixed_precision']:
    weight_dtype = torch.float16

model = BlipDiffusion(sd_pretrained_model_name_or_path=cfg['model']['sd_pretrained_path'], 
                      pretrained_weights_dir=cfg['model']['pretrained_path'],
                      controlnet_pretrained_model_name_or_path=cfg['model']['controlnet_path'],
                      dtype=weight_dtype)


model = model.to(dtype=weight_dtype, device=device)

model.eval()
vis_preprocess = BlipDiffusionInputImageProcessor()
txt_preprocess = BlipCaptionProcessor()

cond_subject = cfg['run']['cond_subject']
tgt_subject = cfg['run']['tgt_subject']
txt_prompt = cfg['run']['txt_prompt']

cond_subjects = [txt_preprocess(cond_subject)]
tgt_subjects = [txt_preprocess(tgt_subject)]
text_prompt = [txt_preprocess(txt_prompt)]

cond_image = Image.open(cfg['run']['cond_image_path']).convert("RGB")

if cfg['run']['cldm_cond_image_path']:
    cldm_cond_image = Image.open(cfg['run']['cldm_cond_image_path']).convert("RGB")
else:
    cldm_cond_image = None

cond_images = vis_preprocess(cond_image).unsqueeze(0).cuda().to(weight_dtype)

samples = {
    "cond_images": cond_images,
    "cond_subject": cond_subjects,
    "tgt_subject": tgt_subjects,
    "prompt": text_prompt,
    "cldm_cond_image": cldm_cond_image
}

num_output = cfg['run']['num_output']

iter_seed = cfg['run']['iter_seed']
guidance_scale = cfg['run']['guidance_scale']
num_inference_steps = cfg['run']['num_inference_steps']
negative_prompt = cfg['run']['negative_prompt']

for i in range(num_output):
    output = model.generate(
        samples,
        seed=iter_seed + i,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        neg_prompt=negative_prompt,
        height=512,
        width=512,
    )

    output[0].save('{}{}_{}.png'.format(cfg['run']['output_dir'], tgt_subject, i))
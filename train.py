from omegaconf import OmegaConf

from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import torch
import torch.backends.cudnn as cudnn

import os
import logging
import math
import time
import datetime
import re
import random
import numpy as np

import webdataset as wds

from common.utils import load_cfg
from common.logger import MetricLogger, SmoothedValue
from common.processors import BlipCaptionProcessor, BlipDiffusionInputImageProcessor, BlipDiffusionTargetImageProcessor
from common.dataset import SubjectDrivenTextToImageDataset
from models.blip_diffusion_models.blip_diffusion import BlipDiffusion


def setup_seeds(config):
    seed = int(config['run']['seed'])

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def main():
    cfg = load_cfg('train_settings.yaml')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    job_id = datetime.now().strftime("%Y%m%d%H%M")[:-1]
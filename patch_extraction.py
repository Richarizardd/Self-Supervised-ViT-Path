### Base Dependencies
import os
import pickle
import sys

### LinAlg / Stats / Plotting Dependencies
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import umap
import umap.plot
from tqdm import tqdm

### Torch Dependencies
import torch
import torch.multiprocessing
import torch.utils.data.dataset as Dataset
from torchvision import transforms
from pl_bolts.models.self_supervised import resnets
from pl_bolts.utils.semi_supervised import Identity
device = torch.device('cuda:0')
torch.multiprocessing.set_sharing_strategy('file_system')

### Model Architectures
from nn_encoder_arch.vision_transformer import vit_small
from nn_encoder_arch.resnet_trunc import resnet50_trunc_baseline


import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class utils:
    BATCH = 512
    IM_SIZE = 224
    
    Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((IM_SIZE, IM_SIZE))
    ])
    
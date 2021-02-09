import torch
import numpy as np
import cv2

def load_img(filepath):
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def torchPSNR(tar_img, prd_img):
    """    Image in range (0,1)    """
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def numpyPSNR(tar_img, prd_img):
    """    Image in range (0,1)    """
    imdff = np.clip(prd_img,0,1) - np.clip(tar_img,0,1)
    rmse = np.sqrt(np.mean(imdff**2))
    ps = 20*np.log10(1/rmse)
    return ps

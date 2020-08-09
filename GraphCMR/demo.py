#!/usr/bin/python
"""
Demo code: given arbitrary input image of at least part of a human, predict mesh

python demo.py --checkpoint=data/models/ours/2020_02_29-18_30_01.pt --img demo/instructions_coffee_0004_00001634.jpg
"""
from __future__ import division
from __future__ import print_function

import torch
from torchvision.transforms import Normalize
import numpy as np
import cv2
import argparse
import json

from utils import Mesh
from models import CMR
from utils.imutils import preprocess_generic
from utils.renderer import Renderer
import config as cfg
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
DEVICE=torch.device("cuda")

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', default=None, help='Path to pretrained checkpoint')
parser.add_argument('--img', type=str, required=True, help='Path to input image')

def process_image(img_file, input_res=224):
    """
    Read image, do preprocessing
    """
    normalize_img = Normalize(mean=cfg.IMG_NORM_MEAN, std=cfg.IMG_NORM_STD)
    rgb_img_in = cv2.imread(img_file)[:,:,::-1].copy().astype(np.float32)
    rgb_img = preprocess_generic(rgb_img_in, input_res)
    disp_img = preprocess_generic(rgb_img_in, input_res, display=True)
    img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
    disp_img = np.transpose(disp_img.astype('float32'),(2,0,1))/255.0
    img = torch.from_numpy(img).float()
    disp_img = torch.from_numpy(disp_img).float()
    norm_img = normalize_img(img.clone())[None]

    return disp_img, norm_img

def disp_imgs(pred_vertices_smpl, mesh, camera_translation, img):
    # Render parametric shape
    img_smpl = renderer.render(pred_vertices_smpl, mesh.faces.cpu().numpy(),
                               camera_t=camera_translation,
                               img=img, use_bg=True, body_color='pink')
    img_smpl2 = renderer.render(pred_vertices_smpl, mesh.faces.cpu().numpy(),
                               camera_t=camera_translation,
                               img=img, use_bg=False, bg_color=(1.0, 1.0, 1.0),
                               body_color='pink')

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    plt.figure(1)
    plt.clf()
    plt.subplot(131)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(img_smpl)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(img_smpl2)
    plt.title('3D mesh')
    plt.axis('off')
    plt.draw()
    plt.show()
    plt.savefig(args.img[:-4]+'_preds'+'.png')

if __name__ == '__main__':
    args = parser.parse_args()
    
    # Load model
    mesh = Mesh(device=DEVICE)
    # Our pretrained networks have 5 residual blocks with 256 channels. 
    # You might want to change this if you use a different architecture.
    model = CMR(mesh, 5, 256, pretrained_checkpoint=args.checkpoint)
    if DEVICE == torch.device("cuda"):
        model.cuda()
    model.eval()

    # Setup renderer for visualization
    renderer = Renderer()

    # Preprocess input image and generate predictions
    img, norm_img = process_image(args.img, input_res=cfg.INPUT_RES)
    if DEVICE == torch.device("cuda"):
        norm_img = norm_img.cuda()
    with torch.no_grad():
        pred_vertices, pred_vertices_smpl, pred_camera, _, _ = model(norm_img)#.cuda())
        
    # Calculate camera parameters for rendering
    camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*cfg.FOCAL_LENGTH/(cfg.INPUT_RES * pred_camera[:,0] +1e-9)],dim=-1)
    camera_translation = camera_translation[0].cpu().numpy()
    pred_vertices = pred_vertices[0].cpu().numpy()
    pred_vertices_smpl = pred_vertices_smpl[0].cpu().numpy()
    img = img.permute(1,2,0).cpu().numpy()
    
    disp_imgs(pred_vertices_smpl, mesh, camera_translation, img)
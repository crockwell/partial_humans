#!/usr/bin/python
from __future__ import print_function
from __future__ import division

import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
import os
import argparse
import json
from collections import namedtuple
from tqdm import tqdm

import config as cfg
from models import CMR, SMPL
from datasets import BaseDataset
from utils.imutils import preprocess_generic
from utils.renderer import Renderer
from utils.mesh import Mesh
from utils.renderer import visualize_reconstruction_custom
from models.geometric_layers import orthographic_projection
import matplotlib.pyplot as plt
import pickle as pkl

import sys
sys.path.insert(0,'../utils')
from calculate_pck import CalcPCK

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Define command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--pretrained_checkpoint', default=None, help='Load a pretrained Graph CNN') 
parser.add_argument('--dataset', default='custom', choices=['vlog','cross_task','instructions','youcook','custom'], help='Choose eval dataset')
parser.add_argument('--crop_setting', default='all', choices=['all','uncropped_keypoint','cropped_keypoint'], help='Choose entire dataset, or cropped or uncropped sets')
parser.add_argument('--pck_eval_threshold', type=float, default=0.5, help='PCK threshold: this number times head size')
parser.add_argument('--num_imgs', type=int, default=0, help='Number of images on which to eval')
parser.add_argument('--eval_pck', default="False", help='if true, eval pck. dataset must be uncropped or cropped keypoint, not all.')
parser.add_argument('--write_imgs', default="False", help='whether to visualize predicted humans')
parser.add_argument('--white_background', default="True", help='if true, visualize predictions on white background. Else bg is image.')
parser.add_argument('--config', default='data/config.json', help='Path to config file containing model architecture etc.')
parser.add_argument('--num_workers', default=1, type=int, help='Number of processes for data loading')
parser.add_argument('--batch_size', default=1, help='Batch size for testing')
parser.add_argument('--img_res', type=int, default=224, help='Rescale bounding boxes to size [img_res, img_res] before feeding it in the network') 
parser.add_argument('--num_channels', type=int, default=256, help='Number of channels in Graph Residual layers') 
parser.add_argument('--num_layers', type=int, default=5, help='Number of residuals blocks in the Graph CNN') 

def write_imgs(input_batch, pred_keypoints_2d_smpl,
                        pred_vertices_smpl, pred_camera, args, renderer):
    """Display predictions"""        
    batch_size = pred_vertices_smpl.shape[0]
    for i in range(min(batch_size, 4)):
        img = input_batch['disp_img'][i].cpu().numpy().transpose(1,2,0)

        # Get LSP keypoints from the full list of keypoints
        to_lsp = list(range(14))
        pred_keypoints_2d_smpl_ = pred_keypoints_2d_smpl[i, to_lsp]

        # Get GraphCNN and SMPL vertices for the particular example
        vertices_smpl = pred_vertices_smpl[i].cpu().numpy()
        cam = pred_camera[i].cpu().numpy()

        # Visualize reconstruction and detected pose
        if args.white_background == "False":
            rend_img_smpl = visualize_reconstruction_custom(img, args.img_res, vertices_smpl, pred_keypoints_2d_smpl_, cam, renderer, use_bg=True)
        else:
            rend_img_smpl = visualize_reconstruction_custom(img, args.img_res, vertices_smpl, pred_keypoints_2d_smpl_, cam, renderer)

        name = input_batch['imgname'][i]
        if args.dataset == 'custom':
            base = cfg.CUSTOM_ROOT
        else:
            base = os.path.join(cfg.BASE_DATA_DIR, args.dataset, args.crop_setting)
        
        preds_type = 'cmr_preds'
        if args.white_background == "False":
            preds_type = 'cmr_preds_overlay'

        path = os.path.join(base, preds_type, name[:-4]+'_preds.png')
        
        try:
            os.makedirs(path)
            os.rmdir(path)
        except:
            print(path,'exists')
            pass

        plt.imsave(path, rend_img_smpl)

def run_evaluation(model, args, dataset, 
                   mesh):
    """Run evaluation on the datasets and metrics we report in the paper. """
        
    # Create SMPL model
    smpl = SMPL().cuda()
    
    # Regressor for H36m joints
    J_regressor = torch.from_numpy(np.load(cfg.JOINT_REGRESSOR_H36M)).float()
    
    # Create dataloader for the dataset
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Transfer model to the GPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    # predictions
    all_kps = {}

    # Iterate over the entire dataset
    for step, batch in enumerate(tqdm(data_loader, desc='Eval', total=args.num_imgs)):
        
        # Get ground truth annotations from the batch
        images = batch['img'].to(device)
        curr_batch_size = images.shape[0]
        
        # Run inference
        with torch.no_grad():
            pred_vertices, pred_vertices_smpl, camera, pred_rotmat, pred_betas = model(images)
            pred_keypoints_3d_smpl = smpl.get_joints(pred_vertices_smpl)
            pred_keypoints_2d_smpl = orthographic_projection(pred_keypoints_3d_smpl, camera.detach())[:, :, :2].cpu().data.numpy()

        eval_part = np.zeros((1,19,2))
        
        # we use custom keypoints for evaluation: MPII + COCO face joints
        # see paper / supplementary for details
        eval_part[0,:14,:] = pred_keypoints_2d_smpl[0][:14]
        eval_part[0,14:,:] = pred_keypoints_2d_smpl[0][19:]

        all_kps[step] = eval_part
        if args.write_imgs == 'True':
            renderer = Renderer(faces=smpl.faces.cpu().numpy())
            write_imgs(batch, pred_keypoints_2d_smpl, pred_vertices_smpl, camera, args, renderer)

    if args.eval_pck == 'True':      
        gt_kp_path = os.path.join(cfg.BASE_DATA_DIR, args.dataset, args.crop_setting, 'keypoints.pkl')
        log_dir = os.path.join(cfg.BASE_DATA_DIR, 'cmr_pck_results.txt')      
        with open(gt_kp_path, 'rb') as f:
            gt = pkl.load(f)
            
        calc = CalcPCK(all_kps, gt, num_imgs=cfg.DATASET_SIZES[args.dataset][args.crop_setting], 
                    log_dir=log_dir, dataset=args.dataset, 
                    crop_setting=args.crop_setting, pck_eval_threshold=args.pck_eval_threshold)
        calc.eval()
            
if __name__ == '__main__':
    args = parser.parse_args()

    # Load model
    mesh = Mesh()
    model = CMR(mesh, args.num_layers, args.num_channels,
                      pretrained_checkpoint=args.pretrained_checkpoint)
    # Setup evaluation dataset
    dataset = BaseDataset(args, args.dataset)
    
    # Run evaluation
    run_evaluation(model, args, dataset, mesh)

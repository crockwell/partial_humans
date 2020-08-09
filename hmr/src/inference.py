"""
HMR inference.
From an image input, model outputs 85D latent vector
consisting of [cam (3 - [scale, tx, ty]), pose (72), shape (10)]
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .models import get_encoder_fn_separate

from .tf_smpl.batch_lbs import batch_rodrigues
from .tf_smpl.batch_smpl import SMPL
from .tf_smpl.projection import batch_orth_proj_idrot
from .tf_smpl import projection as proj_util
import math
import pickle as pkl

from tensorflow.python.ops import control_flow_ops

import tensorflow as tf
import numpy as np

import deepdish as dd
import os
import matplotlib.pyplot as plt

from tqdm import tqdm

# For drawing
from .util import renderer as vis_util

# for eval
import sys
sys.path.insert(0,'../utils')
from calculate_pck import CalcPCK

class HMRInference(object):
    def __init__(self, config, data_loader):
        self.config = config

        self.data_format = config.data_format
        self.smpl_model_path = config.smpl_model_path
        self.load_path = config.load_path
        self.num_imgs = config.num_imgs

        # Data size
        self.img_size = config.img_size
        self.num_stage = config.num_stage
        self.batch_size = config.batch_size

        self.num_cam = 3
        self.proj_fn = batch_orth_proj_idrot

        self.num_theta = 72  # 24 * 3
        self.total_params = self.num_theta + self.num_cam + 10

        self.num_itr_per_epoch = 5000 / self.batch_size

        # First make sure data_format is right
        if self.data_format == 'NCHW':
            # B x H x W x 3 --> B x 3 x H x W
            data_loader['image'] = tf.transpose(data_loader['image'],
                                                [0, 3, 1, 2])

        self.image_loader = data_loader['image']
        
        self.display_image_loader = data_loader['display_image']
        self.filename = data_loader['filename']

        self.global_step = tf.Variable(0, name='global_step', trainable=False)

        # For visualization:
        num2show = np.minimum(8, self.batch_size)
        self.show_these = tf.constant(np.hstack([np.arange(num2show)]), tf.int32)
        self.write_imgs = (config.write_imgs == 'True')
        self.white_background = (config.white_background == 'True')
        out_folder = 'preds'
        if not self.white_background:
            out_folder = 'preds_overlay'
        self.dataset = config.dataset
        self.crop_setting = config.crop_setting
        base_folder = None
        if config.custom_dir:
            base_folder = config.custom_dir
            self.log_dir = None
        else:
            base_folder = os.path.join(config.base_dir, self.dataset, self.crop_setting)
            self.log_dir = os.path.join(config.base_dir, config.logdir)
        self.out_dir = os.path.join(base_folder, out_folder)

        # for testing
        self.eval_pck = (config.eval_pck == 'True')
        self.pck_eval_threshold = config.pck_eval_threshold
        self.gt_kp_path = os.path.join(base_folder, 'keypoints.pkl')
            
        # Weight decay
        self.e_wd = config.e_wd
        self.size = data_loader['size']

        # Instantiate SMPL
        self.smpl = SMPL(self.smpl_model_path)
        
        self.idx = data_loader['idx']
        self.build_model_inference(self.image_loader, self.display_image_loader)
        
        self.pre_train_saver = tf.train.Saver()

        def load_pretrain(sess):
            self.pre_train_saver.restore(sess, self.load_path)

        init_fn = load_pretrain

        self.sv = tf.train.Supervisor(
            global_step=self.global_step,
            init_fn=init_fn)
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess_config = tf.ConfigProto(
            allow_soft_placement=False,
            log_device_placement=False,
            gpu_options=gpu_options)

        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        print("[*] MODEL dir: %s" % self.load_path)

    def load_mean_param(self):
        mean = np.zeros((1, self.total_params))
        # Initialize scale at 0.9
        mean[0, 0] = 0.9
        mean_path = os.path.join(
            os.path.dirname(self.smpl_model_path), 'neutral_smpl_mean_params.h5')
        mean_vals = dd.io.load(mean_path)

        mean_pose = mean_vals['pose']
        # Ignore the global rotation.
        mean_pose[:3] = 0.
        mean_shape = mean_vals['shape']

        # This initializes the global pose to be up-right when projected
        mean_pose[0] = np.pi

        mean[0, 3:] = np.hstack((mean_pose, mean_shape))
        mean = tf.constant(mean, tf.float32)
        self.mean_var = tf.Variable(
            mean, name="mean_param", dtype=tf.float32, trainable=True)
        init_mean = tf.tile(self.mean_var, [self.batch_size, 1])
        return init_mean

    def build_model_inference(self, image, display_image):
        all_verts = []
        all_pred_kps = []
        all_pred_cams = []
        theta_here = None
        
        img_enc_fn, threed_enc_fn = get_encoder_fn_separate()

        reuse=False
        self.mean_params = self.load_mean_param()
        theta_prev = self.mean_params
        img_feat, _ = img_enc_fn(
                image, 
                reuse=reuse, is_training=False) # 2048 feats
        
        for i in np.arange(self.num_stage):            
            state = tf.concat([img_feat, theta_prev], 1)
            delta_theta = None
            if i == 0:
                delta_theta, _ = threed_enc_fn(
                            state, num_output=self.total_params,
                            reuse=reuse, is_training=False)
            else:
                delta_theta, _ = threed_enc_fn(
                            state, num_output=self.total_params, 
                            reuse=True, is_training=False)
            theta_here = theta_prev + delta_theta
            theta_prev = theta_here
            
            cams = theta_here[:, :self.num_cam]
            poses = theta_here[:, self.num_cam:(self.num_cam + self.num_theta)]
            shapes = theta_here[:, (self.num_cam + self.num_theta):]
            verts, Js, pred_Rs = self.smpl(shapes, poses, get_skin=True)
            pred_kp = batch_orth_proj_idrot(
                Js, cams, name='proj2d_stage%d' % i)
            if i == self.num_stage - 1:
                all_verts.append(tf.gather(verts, self.show_these))
                all_pred_kps.append(tf.gather(pred_kp, self.show_these))
                all_pred_cams.append(tf.gather(cams, self.show_these))
        
        self.all_verts = tf.stack(all_verts, axis=1)
        self.all_kps = tf.stack(all_pred_kps, axis=1)
        self.all_cams = tf.stack(all_pred_cams, axis=1)
        self.imgs = tf.gather(image, self.show_these)
        self.disp_imgs = tf.gather(display_image, self.show_these)
        self.theta = theta_here

    def visualize_img(self, disp_img, vert, pred_kp, cam, renderer):
        """
        Overlays gt_kp and pred_kp on img.
        Draws vert with text.
        Renderer is an instance of SMPLRenderer.
        """
        # Fix a flength so i can render this with persp correct scale
        f = 5.
        tz = f / cam[0]
        cam_for_render = 0.5 * self.img_size * np.array([f, 1, 1])
        cam_t = np.array([cam[1], cam[2], tz])
        if self.white_background:
            rend_img = renderer(vert + cam_t, cam_for_render)
        else:
            # Undo pre-processing.
            disp_img = (disp_img + 1) * 0.5
            rend_img = renderer(vert + cam_t, cam_for_render, img=disp_img)
            
        return rend_img 
    
    def draw_results(self, result):

        # This is B x H x W x 3
        disp_imgs = result["disp_img"]
        # B x 19 x 3
        if self.data_format == 'NCHW':
            disp_imgs = np.transpose(disp_imgs, [0, 2, 3, 1])
        # This is B x 1 x 6890 x 3
        est_verts = result["e_verts"]
        # B x 1 x 19 x 2
        joints = result["joints"]
        # B x 1 x 3
        cams = result["cam"]
        filename = result['filename']

        img_summaries = []
        
        for img_id, (disp_img, verts, joints, cams, fname) in enumerate(
                zip(disp_imgs, est_verts, joints, cams, filename)):

            combined_pred = self.visualize_img(disp_img, verts[0], joints[0], cams[0], self.renderer)
            imgs = np.concatenate(((disp_img+1)*0.5, combined_pred/255.0), axis=1)
            
            if self.white_background:
                path = os.path.join(self.out_dir, fname[:-3]+'png')
            else:
                path = os.path.join(self.out_dir, fname[:-3]+'png')
            try:
                os.makedirs(path)
                os.rmdir(path)
                plt.imsave(path, imgs)
            except:
                pass

    def run(self):
        # For rendering!
        self.renderer = vis_util.SMPLRenderer(
            img_size=self.img_size,
            face_path=self.config.smpl_face_path)
        
        step = 0
        
        all_kps = {}
               
        with tqdm(total=self.num_imgs, desc='Making Predictions') as pbar:
            with self.sv.managed_session(config=self.sess_config) as sess:
                while not self.sv.should_stop():                
                    fetch_dict = {
                        "disp_img": self.disp_imgs,
                        "e_verts": self.all_verts,
                        "joints": self.all_kps,
                        "cam": self.all_cams,
                        'idx': self.idx,
                        'filename': self.filename,
                        'imgs': self.imgs,
                        }
                    result = sess.run(fetch_dict)
                    if self.write_imgs:
                        self.draw_results(result)
                    all_kps[int(result['idx'][0])] = result['joints'][0]
                    
                    epoch = float(step) / self.num_itr_per_epoch
                        
                    if len(all_kps) == self.num_imgs:
                        self.sv.request_stop()
                    elif len(all_kps) > self.num_imgs:
                        print('seems num_imgs is smaller than size of full dataset')
                        assert(False)

                    step += 1
                    pbar.update(step-pbar.n)
        
        if self.eval_pck:
            with open(self.gt_kp_path, 'rb') as f:
                gt = pkl.load(f)
                
            calc = CalcPCK(all_kps, gt,
                        self.num_imgs, self.log_dir, 
                        self.dataset, self.crop_setting, 
                        self.pck_eval_threshold)
            calc.eval()
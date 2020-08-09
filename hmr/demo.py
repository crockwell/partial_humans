"""
Demo of HMR.

Sample usage:

# On images of at least part of a person
python -m demo --img_path demo/vlog_q_u_Q_v_qNSfZz0HquQ_017_frame000151.jpg
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '4'

def visualize(img, proc_param, joints, verts, cam, img_path):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    centered_cam = (0.9, 0, 0) # used to see fully-visible views of person
    cam_for_render2, vert_shifted2, joints_orig2 = vis_util.get_original(
        proc_param, verts, centered_cam, joints, img_size=img.shape[:2])
    

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints)

    f = 5.
    tz = f / cam[0]
    cam_for_render = 0.5 * 224 * np.array([f, 1, 1])
    cam_t = np.array([cam[1], cam[2], tz])
    rend_img_overlay = renderer(
        verts + cam_t, cam=cam_for_render, img=img)
    #rend_img_overlay = renderer(
    #    vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        verts + cam_t, cam=cam_for_render, img_size=img.shape[:2])
    cam_t2 = np.array([cam[1], cam[2], tz*2])
    rend_img_vp1 = renderer.rotated(
        verts + cam_t2, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        verts + cam_t2, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib
    matplotlib.use('agg')
    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(131)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.draw()
    plt.show()
    plt.savefig(img_path[:-4]+'_preds'+'.png')
    # import ipdb
    # ipdb.set_trace()


def preprocess_image(img_path):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if np.max(img.shape[:2]) != config.img_size:
        print('Resizing so the max image size is %d..' % config.img_size)
        scale = (float(config.img_size) / np.max(img.shape[:2]))
    else:
        scale = 1.
    center = np.round(np.array(img.shape[:2]) / 2).astype(int)
    # image center in (x,y)
    center = center[::-1]

    crop, proc_param, disp_img = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, disp_img


def main(img_path):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    visualize(img, proc_param, joints[0], verts[0], cams[0], img_path)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path)

"""
TF util operations.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math


def keypoint_l1_loss(kp_gt, kp_pred, scale=1., name=None):
    """
    computes: \Sum_i [0.5 * vis[i] * |kp_gt[i] - kp_pred[i]|] / (|vis|)
    Inputs:
      kp_gt  : N x K x 3
      kp_pred: N x K x 2
    """
    with tf.name_scope(name, "keypoint_l1_loss", [kp_gt, kp_pred]):
        kp_gt = tf.reshape(kp_gt, (-1, 3))
        kp_pred = tf.reshape(kp_pred, (-1, 2))

        vis = tf.expand_dims(tf.cast(kp_gt[:, 2], tf.float32), 1)
        res = tf.losses.absolute_difference(kp_gt[:, :2], kp_pred, weights=vis)
        return res
    
def smpl_loss_l1(smpl_gt, smpl_pred, name=None, num_buckets=None):
    """
    computes: unweighted loss of SMPL rotation params, L1
    Inputs:
      smpl_gt  : N x 85
      smpl_pred: N x 85
    """
    with tf.name_scope(name, "smpl_loss_l1", [smpl_gt, smpl_pred]):
        res = tf.losses.absolute_difference(smpl_gt, smpl_pred)
        return res
    
def smpl_loss_l2(smpl_gt, smpl_pred, name=None, num_buckets=None):
    """
    computes: unweighted loss of SMPL rotation params, L2
    Inputs:
      smpl_gt  : N x 85
      smpl_pred: N x 85
    """
    with tf.name_scope(name, "smpl_loss_l2", [smpl_gt, smpl_pred]):
        res = tf.losses.mean_squared_error(smpl_gt, smpl_pred)
        return res

'''
the lack of logits makes it too variable for fair comparison -- really just 
luck of the draw whether happens to bucket close to prediction
def smpl_loss_quantized_regression(smpl_gt, smpl_pred, num_buckets=32, name=None):
    """
    so we could calculate quantized loss in regression training scenario
    computes: unweighted loss of quantized SMPL rotation params 
    compared to one-hot encoding
    Inputs:
      smpl_gt  : N x 85
      smpl_pred: N x 85
    """
    pi = tf.constant(math.pi)
    bucket_size = 2*pi/num_buckets
    gt_bucketed = tf.cast((smpl_gt + pi)/bucket_size, tf.int32)
    onehot_gt = tf.one_hot(gt_bucketed, num_buckets)
    pred_bucketed = tf.cast((smpl_pred + pi)/bucket_size, tf.int32)
    onehot_pred = tf.one_hot(pred_bucketed, num_buckets)
    with tf.name_scope(name, "smpl_loss_quantized_regression", [onehot_gt, onehot_pred]):
        res = tf.losses.softmax_cross_entropy(onehot_labels=onehot_gt, logits=onehot_pred)
        return res
'''
    
def smpl_loss_quantized(smpl_gt, smpl_pred, num_buckets=32, name=None):
    """
    calculates loss for optimization in logits training scenario
    computes: unweighted loss of quantized SMPL rotation params 
    compared to one-hot encoding
    Inputs:
      smpl_gt  : N x 85
      smpl_pred: N x 85 X num_buckets
    """
    pi = tf.constant(math.pi)
    bucket_size = 2*pi/num_buckets
    gt_bucketed = tf.cast((smpl_gt + pi)/bucket_size, tf.int32)
    onehot_gt = tf.one_hot(gt_bucketed, num_buckets)
    with tf.name_scope(name, "smpl_loss_quantized", [smpl_gt, smpl_pred]):
        res = tf.losses.softmax_cross_entropy(onehot_labels=onehot_gt, logits=smpl_pred)
        return res
    
def smpl_loss_clustering(smpl_gt, smpl_pred, num_clusters=32, name=None):
    pass

def compute_accuracy(smpl_gt, smpl_pred, threshold=math.pi/16, name=None):
    """
    computes: accuracy of smpl joints
    Inputs:
      threshold: how close it has to be to be correct
      smpl_gt  : N x 85
      smpl_pred: N x 85
    """
    part_ref = {'ankle':[8,7],'knee':[5,4],'hip':[2,1],
            'wrist':[21,20],'elbow':[19,18],'shoulder':[17,16],
            'head':[15,12],'stomach':[9,6,3,0], 'chest':[14,13], 
            'foot':[11,10],'hand':[23,22], 
            'cam': [-1], 'shape params': [25], 'orientation': [0]}

    differences = tf.maximum(smpl_gt, smpl_pred) - tf.minimum(smpl_gt, smpl_pred)
        
    correct = tf.cast(tf.math.less_equal(differences, threshold), tf.float32)
    accuracy = {}
    accuracy['joints, cams, shape'] = tf.reduce_mean(correct)
    accuracy['all joints'] = tf.reduce_mean(correct[:, 6:75])
    for part in part_ref:
        accuracy[part] = 0
        ct = 0
        for num in part_ref[part]:
            strt = (num+1)*3
            fin = (num+2)*3
            accuracy[part] += tf.reduce_mean(correct[:,strt:fin])
            ct += 1
        accuracy[part] /= ct
    return accuracy
    
def compute_3d_loss(params_pred, params_gt, has_gt3d):
    """
    Computes the l2 loss between 3D params pred and gt for those data that has_gt3d is True.
    Parameters to compute loss over:
    3Djoints: 14*3 = 42
    rotations:(24*9)= 216
    shape: 10
    total input: 226 (gt SMPL params) or 42 (just joints)

    Inputs:
      params_pred: N x {226, 42}
      params_gt: N x {226, 42}
      # has_gt3d: (N,) bool
      has_gt3d: N x 1 tf.float32 of {0., 1.}
    """
    with tf.name_scope("3d_loss", [params_pred, params_gt, has_gt3d]):
        weights = tf.expand_dims(tf.cast(has_gt3d, tf.float32), 1)
        res = tf.losses.mean_squared_error(
            params_gt, params_pred, weights=weights) * 0.5
        return res


def align_by_pelvis(joints):
    """
    Assumes joints is N x 14 x 3 in LSP order.
    Then hips are: [3, 2]
    Takes mid point of these points, then subtracts it.
    """
    with tf.name_scope("align_by_pelvis", [joints]):
        left_id = 3
        right_id = 2
        pelvis = (joints[:, left_id, :] + joints[:, right_id, :]) / 2.
        return joints - tf.expand_dims(pelvis, axis=1)

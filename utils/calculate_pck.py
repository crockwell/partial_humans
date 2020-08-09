from __future__ import print_function
from tqdm import tqdm
import numpy as np
import copy

class CalcPCK(object):
    def __init__(self, preds, gt, num_imgs, log_dir, dataset, crop_setting, pck_eval_threshold, preds_in_pixels=False):
         ''' Inputs:
        preds: formatted as dict, mapping image id to predictions such that:
            -Image id corresponds to line number in "images.txt"
            -either
                -predictions are in terms of the uncropped image pixels (set preds_in_pixels=True)
                -predictions are formatted so that [-1, 1] is a square around the image. (preds_in_pixels=False)
                    Both HMR and CMR preprocess images this way, so this is the format we use.
                    In detail, they preprocess such that the entire original image 
                    is visible, then padded to be a square 224x224 input, and predictions
                    correspond to this space.
        gt: formatted as dict, mapping image id to a number of ground truth values. 
            -keypoints in uncropped image, with visibility score. 0 is not labeled,
                1 is labeled and visible in cropped image,
                2 is labeled but not visible in cropped image.
            -uncropped image size
            -scale (for PCK calculation)
            -cropped image size (used for transforming predictions in cropped image
                to uncropped setting)
            -image path
        num_imgs: number of images in eval set (int)
        log_dir: directory to write scores
        dataset: vlog, cross_task, instructions, or youcook
        crop_setting: "cropped_keypoint" or "uncropped_keypoint". Note eval is not available on
            "all", since there is no head available on some images, used to calculate "scale".
        pck_eval_threshold: number of times head size. We use 0.5.
        preds_in_pixels: if original predictions are already in pixels, we do not need
            to call the "convert_preds_to_pixels" function
         '''
         self.preds = preds
         self.gt = gt
         self.is_dataset_cropped = (crop_setting == 'cropped_keypoint')
         self.num_imgs = num_imgs
         self.log_dir = log_dir
         self.dataset = dataset
         self.crop_setting = crop_setting
         self.pck_eval_threshold = pck_eval_threshold
         self.preds_in_pixels = preds_in_pixels

    def convert_preds_to_pixels(self, pred, gt):
        '''
        ground truths are labeled based on original images,
        which are typically larger than 224x224 and are not padded.
        Our predictions are in [-1,1], and non-square images
        were padded to be square. We convert to ground truth pixel space.
        '''
        size = (gt['uncropped_width'], gt['uncropped_height'])
        if self.is_dataset_cropped:
            size = (gt['cropped_width'], gt['cropped_height'])
        larger_size_dim = np.argmax(size) 
        smaller_size_dim = 0
        if larger_size_dim == 0:
            smaller_size_dim = 1

        pred = (pred[0]+1)*.5*size[larger_size_dim]
        padding = (size[larger_size_dim]-size[smaller_size_dim])/2.0
        pred[:,smaller_size_dim] -= padding # w, h

        if self.is_dataset_cropped:
            pred[:, 0] += gt['pixels_cropped_left']
            pred[:, 1] += gt['pixels_cropped_above']

        return pred

    def img_pck(self, pred, gt, scale, pck):
        if not self.preds_in_pixels:
            pred = self.convert_preds_to_pixels(pred, gt)

        pck_image = copy.deepcopy(pck)
        for metric in pck_image:
            pck_image[metric]['Correct'], pck_image[metric]['Count'] = 0, 0

        for j in range(19):
            if gt['kps'][j][0] > 0:
                dist = np.linalg.norm(pred[j]-gt['kps'][j,:2])
                vis = None
                if self.is_dataset_cropped:
                    if gt['kps'][j,2] == 1:
                        vis = 'Visible'
                    elif gt['kps'][j,2] == 2:
                        vis = 'Not Visible'

                if dist < self.pck_eval_threshold * scale:
                    pck_image['All']['Correct'] += 1
                    if vis is not None:
                        pck_image[vis]['Correct'] += 1

                pck_image['All']['Count'] += 1
                if vis is not None:
                    pck_image[vis]['Count'] += 1

        for metric in pck:
            if pck_image[metric]['Count'] > 0:
                pck[metric]['Correct'] += float(pck_image[metric]['Correct']) / pck_image[metric]['Count']
                pck[metric]['Count'] += 1

        return pck

    def eval(self):
        pck = {'All': {'Correct': 0, 'Count': 0}}
        if self.is_dataset_cropped:
            pck['Visible'] = {'Correct': 0, 'Count': 0}
            pck['Not Visible'] = {'Correct': 0, 'Count': 0}

        with tqdm(total=self.num_imgs, desc='Evaluating') as pbar:
            for idx in self.gt:
                pbar.update(1)
                try:
                    scale = self.gt[idx]['scale']
                except:
                    continue
                if scale > 0: # so head is annotated
                    '''
                    Note: we calculate image PCK per image, and average across images.
                        Not doing so gives significantly more weight to images with more keypoints
                        visible compared to less, which we don't want.
                    '''
                    pck = self.img_pck(self.preds[idx], self.gt[idx], scale, pck)

        with open(self.log_dir, 'a+') as f:
            text = self.dataset + ' ' + self.crop_setting
            print(text)
            print(text, file=f)
            for metric in pck:
                text = metric + ' Keypoints, % Correct @ 0.5: ' + \
                        str(round(pck[metric]['Correct']/pck[metric]['Count']*100,1)) + \
                            ', Count:' + str(pck[metric]['Count']) + ' Images'
                print(text)
                print(text, file=f)
            print('', file=f)
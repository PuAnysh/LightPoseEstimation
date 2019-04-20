import os
import cv2
import random
import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from pycocotools.coco import COCO
from dataloader.cocodataset.coco_process_utils import *
from dataloader.cocodataset.process_utils import *
import torch
MIN_KEYPOINTS = 5
MIN_AREA = 32 * 32

class COCOBBox():
    def __init__(self, cfg, train=True):
        self.coco_year = 2017
        self.cfg = cfg
        if train:
            self.cocobbox = COCO(cfg.Instances_ANNOTATION_train_DIR)
            #self.cocokypt = COCO(cfg.keyPoints_ANNOTATION_train_DIR)
            self.data_path = cfg.IMG_DIR
            self.do_augment = train
            self.indices = self._clean_annot(self.cocobbox , 'train')
            self.input_size = cfg.INPUT_SIZE
            self.HEAT_MAP_SIZE = cfg.HEAT_MAP_SIZE
            self.sigmaHM = cfg.sigmaHM
            self.sigma_limb = cfg.sigma_limb
            self.seq = self.__getIAAseq__(cfg.INPUT_SIZE)
            self.seq_size = self.__getIAAseq_scale__(cfg.INPUT_SIZE)
            #print('Loaded {} images for {}'.format(len(self.indices), 'train'))

        # load annotations that meet specific standards
        self.img_dir = cfg.IMG_DIR
        #print('Loaded {} images for {}'.format(len(self.indices), split))
    def bbox2imgbbox(self , image , bbox_):
        bbox = []
        for row in bbox_:
            bbox.append(ia.BoundingBox(x1 = row[0] ,y1 = row[1] ,x2 = row[0]+row[2] , y2= row[1]+row[3]))
        return ia.BoundingBoxesOnImage(bbox , shape=image.shape)

    def imgbbox2bbox(self,bbox_aug):
        bbox = []
        for row in bbox_aug.bounding_boxes:
            tmp_bbox = [row.x1 , row.y1 , row.x2 , row.y2]
            bbox.append(tmp_bbox)
        return np.array(bbox)

    def pose2keypoints(self, image, pose):
        keypoints = []
        for row in range(int(pose.shape[0])):
            x = pose[row,0]
            y = pose[row,1]
            keypoints.append(ia.Keypoint(x=x, y=y))
        return ia.KeypointsOnImage(keypoints, shape=image.shape)

    def keypoints2pose(self, keypoints_aug):
        one_person = []
        for kp_idx, keypoint in enumerate(keypoints_aug.keypoints):
            x_new, y_new = keypoint.x, keypoint.y
            one_person.append(np.array(x_new).astype(np.float32))
            one_person.append(np.array(y_new).astype(np.float32))
        return np.array(one_person).reshape([-1,2])

    def __getIAAseq__(self,size):
        seq = iaa.Sequential(
            [
                # Apply the following augmenters to most images.

                # iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False),
                #
                iaa.Affine(
                    # scale={"x": (0.75, 1.25), "y": (0.75, 1.25)},
                    # translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                    rotate=(-5, 5),
                    shear=(-2, 2),
                    order=[0, 1],
                    cval=0,
                    mode="constant"
                )

            ]
        )
        # augmentation choices
        # seq_det = seq.to_deterministic()
        return seq

    def __getIAAseq_scale__(self, size):
        seq = iaa.Sequential(
            [
                # Apply the following augmenters to most images.
                # iaa.CropAndPad(percent=(-0.25, 0.25), pad_mode=["edge"], keep_size=False),
                #
                iaa.Scale(size,
                          interpolation='nearest',
                          name=None,
                          deterministic=False,
                          random_state=None)
            ]
        )
        # augmentation choices
        # seq_det = seq.to_deterministic()
        return seq



    def __getAug__(self , img , keypoint , heat_maps , limb_sigma ,  mask):
        seq_det = self.seq.to_deterministic()
        img = seq_det.augment_image(img)
        mask = seq_det.augment_image(mask)
        heat = []
        limbs = []
        for i in range(heat_maps.shape[0]):
            heat_map = heat_maps[i]
            heat_map = seq_det.augment_image(heat_map)
            heat.append(heat_map)
        heat = np.array(heat)
        for i in range(limb_sigma.shape[0]):
            limb = limb_sigma[i]
            limb = seq_det.augment_image(limb)
            limbs.append(limb)
        limbs = np.array(limbs)
        _keypoint = self.pose2keypoints(img , keypoint)
        _keypoint = seq_det.augment_keypoints(_keypoint)
        keypoint[:,0:2] = self.keypoints2pose(_keypoint)


        return  img , keypoint , heat , limbs ,  mask

    def __getitem__(self, index):
        index = self.indices[index]
        anno_ids = self.cocobbox.getAnnIds(index)
        annots = self.cocobbox.loadAnns(anno_ids)
        annots = list(filter(lambda annot: check_annot(annot), annots))
        img_path = os.path.join(self.data_path, self.cocobbox.loadImgs([index])[0]['file_name'])
        #print(img_path)
        img = cv2.imread(img_path)
        img = img.astype('float32') / 255.
        keypoints , bboxes = get_keypoints_bbox(self.cocobbox, img, annots)
        ignore_mask = get_ignore_mask(self.cocobbox, img, annots)

        idx =  random.randint(0,1007)%(keypoints.shape[0])
        y1, x1, y2, x2 = bboxes[idx].astype(np.int32)
        x2 = x2 + x1
        y2 = y2 + y1
        keypoint = keypoints[idx]
        for i in range(keypoint.shape[0]):
            keypoint[i, 0] -= y1
            keypoint[i, 1] -= x1
        img = img[x1:x2 , y1:y2 , :]
        ignore_mask = ignore_mask[x1:x2 , y1:y2]
        heat_map = get_heatmap(self.cocobbox, img, keypoint.reshape(-1,18,3), self.sigmaHM)
        limb_sigma = get_limbSigma(self.cocobbox, img, keypoint.reshape(-1,18,3), self.sigma_limb)
        #heat_map = heat_map[: , x1:x2 , y1:y2]
        #limb_sigma = limb_sigma[: , x1:x2 , y1:y2]
        img , keypoint , heat_map , limb_sigma, ignore_mask = self.__getAug__(img , keypoint , heat_map , limb_sigma, ignore_mask)
        # resize
        img = cv2.resize(img , (self.input_size , self.input_size)).astype(np.float32)
        heat_map = cv2.resize(heat_map.transpose(1 ,2 ,0) , (self.HEAT_MAP_SIZE , self.HEAT_MAP_SIZE)).transpose(2,0,1).astype(np.float32)
        limb_sigma = cv2.resize(limb_sigma.transpose(1 ,2 ,0) , (self.HEAT_MAP_SIZE , self.HEAT_MAP_SIZE)).transpose(2,0,1).astype(np.float32)
        ignore_mask = cv2.resize(ignore_mask , (self.HEAT_MAP_SIZE , self.HEAT_MAP_SIZE)).astype(np.float32)
        scale_x = self.input_size/(x2-x1)
        scale_y = self.input_size / (y2 - y1)
        keypoint[:,0] *= scale_y
        keypoint[:, 1] *= scale_x
        img = normalize(img)
        img = torch.from_numpy(img.astype(np.float32))
        heat_map = torch.from_numpy(heat_map.astype(np.float32))
        limb_sigma = torch.from_numpy(limb_sigma.astype(np.float32))
        ignore_mask = torch.from_numpy(ignore_mask.astype(np.float32))
        bbox = torch.from_numpy(bboxes[idx].reshape(-1,4).astype(np.float32))

        return  img , keypoint , bbox , heat_map , limb_sigma , ignore_mask


    def _clean_annot(self,coco,split):
        # 返回包含有效人体的图片index,包括清理一下垃圾的数据
        #print('Filtering annotations for {}'.format(split))
        person_ids = coco.getCatIds(catNms=['person'])
        indices_tmp = sorted(coco.getImgIds(catIds=person_ids))
        indices = np.zeros(len(indices_tmp))
        valid_count = 0
        for i in range(len(indices_tmp)):
            anno_ids = coco.getAnnIds(indices_tmp[i])
            annots = coco.loadAnns(anno_ids)
            # Coco standard constants
            annots = list(filter(lambda annot: check_annot(annot), annots))
            if len(annots) > 0:
                indices[valid_count] = indices_tmp[i]
                valid_count += 1
        indices = indices[:valid_count]
        return indices

    def __len__(self):
        return len(self.indices)

def visualize_heatmap(img, heat_maps, displayname = 'heatmaps'):
    heat_maps = heat_maps.max(axis=0)
    heat_maps = (heat_maps/heat_maps.max() * 255.).astype('uint8')
    heat_maps = cv2.resize(heat_maps, (img.shape[0] , img.shape[1]))
    img = img.copy()
    colored = cv2.applyColorMap(heat_maps, cv2.COLORMAP_JET)
    #img = img*0.1+colored*0.8
    img = img * 0.5 + colored * 0.5
    #img = heat_maps
    #img = cv2.addWeighted(img, 0.6, colored, 0.4, 0)
    cv2.imshow(displayname, heat_maps)
    cv2.waitKey()

def visualize_keypoints(img, keypoints, body_part_map):
    img = img.copy()
    keypoints = keypoints.astype('int32')
    for i in range(keypoints.shape[0]):
        x = keypoints[i, 0]
        y = keypoints[i, 1]
        if keypoints[i, 2] > 0:
            cv2.circle(img, (x, y), 3, (0, 1, 0), -1)
    for part in body_part_map:
        keypoint_1 = keypoints[part[0]]
        x_1 = keypoint_1[0]
        y_1 = keypoint_1[1]
        keypoint_2 = keypoints[part[1]]
        x_2 = keypoint_2[0]
        y_2 = keypoint_2[1]
        if keypoint_1[2] > 0 and keypoint_2[2] > 0:
            cv2.line(img, (x_1, y_1), (x_2, y_2), (1, 0, 0), 2)
    cv2.imshow('keypoints', img)
    cv2.waitKey()

if __name__ == '__main__':
    import config as cfg
    import time
    data = COCOBBox(cfg)

    while(True):
        s = time.time()
        idx = random.randint(1,100007)%30000
        img , keypoints , bboxes , heat_maps , limb_sigma , ignore_mask = data.__getitem__(idx)
        print(img.shape)
        print(keypoints.shape)
        print(bboxes.shape)
        print(heat_maps.shape)
        print(limb_sigma.shape)
        e = time.time()
        print('time :{}'.format(e - s))
        img = denormalize(img.cpu().numpy())
        visualize_keypoints(img, keypoints.reshape(18,3), BODY_PARTS)
        #for i in range(17):
        #    heat = limb_sigma[i,:,:].cpu().numpy()
        #    heat = cv2.resize(heat , (img.shape[0] , img.shape[1]))
        #    cv2.imshow('234',heat*255)
        #    cv2.waitKey()



        #visualize_keypoints(img, keypoints, BODY_PARTS)
        visualize_heatmap(img, heat_maps.cpu().numpy())
        visualize_heatmap(img, limb_sigma.cpu().numpy())


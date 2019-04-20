import os
import cv2
import pickle
import numpy as np
from dataloader.cocodataset.process_utils import DrawGaussian

# COCO typical constants
MIN_KEYPOINTS = 5
MIN_AREA = 128 * 128

# Non traditional body parts
BODY_PARTS = [
    (0,1),   # nose - left eye
    (0,2),   # nose - right eye
    (1,3),   # left eye - left ear
    (2,4),   # right eye - right ear
    (17,5),  # neck - left shoulder
    (17,6),  # neck - right shoulder
    (5,7),   # left shoulder - left elbow
    (6,8),   # right shoulder - right elbow
    (7,9),   # left elbow - left hand
    (8,10),  # right elbow - right hand
    (17,11), # neck - left waist
    (17,12), # neck - right waist
    (11,13), # left waist - left knee
    (12,14), # right waise - right knee
    (13,15), # left knee - left foot
    (14,16), # right knee - right foot
    (0,17)   # nose - neck
]

FLIP_INDICES = [(1,2), (3,4), (5,6), (7,8), (9,10), (11,12), (13,14), (15,16)]
FLIP_INDICES_PAF = [(0,1), (2,3), (4,5), (6,7), (8,9), (10,11), (12,13), (14,15)]


def check_annot(annot):
    #保证注释内容的有效性，大于MIN_KEYPOINTS，大小大于MIN_AREA等特性
    return annot['num_keypoints'] >= MIN_KEYPOINTS and annot['area'] > MIN_AREA and not annot['iscrowd'] == 1


def get_heatmap(coco, img, keypoints, sigma):
    #获得热力图
    #获得点数，[人数,关键点数,三维数据]
    n_joints = keypoints.shape[1]
    out_map = np.zeros((n_joints, img.shape[0], img.shape[1]))
    for person_id in range(keypoints.shape[0]):
        keypoints_person = keypoints[person_id]
        # 迭代每一个类型的关键点，组成高斯成像图
        for i in range(keypoints.shape[1]):
            keypoint = keypoints_person[i]
            # Ignore unannotated keypoints
            # 0代表未标注
            if keypoint[2] > 0:
                x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                x = x.astype(np.float64)
                y = y.astype(np.float64)
                x -= keypoint[0]
                y -= keypoint[1]
                distance = np.sqrt(x * x + y * y)
                distance = 1/(2*3.14*sigma*sigma) * np.exp(-distance/(2*sigma*sigma))
                #distance = distance/distance.max()
                out_map[i] = np.maximum(out_map[i], distance)
    #out_map[n_joints] = 1 - np.sum(out_map[0:n_joints], axis=0) # Last heatmap is background
    return out_map

def get_limbSigma(coco, img, keypoints, sigma_limb):
    out_limbSigma = np.zeros((len(BODY_PARTS) , img.shape[0] , img.shape[1]))
    for person_id in range(keypoints.shape[0]):
        keypoints_person = keypoints[person_id]
        for i in range(len(BODY_PARTS)):
            part = BODY_PARTS[i]
            # 取下第一个点的X,Y坐标
            keypoint_1 = keypoints_person[part[0], :2]
            # 取下第二个点的X,Y坐标
            keypoint_2 = keypoints_person[part[1], :2]
            # 保证是有效的点（0代表未标注）
            if keypoints_person[part[0], 2] > 0 and keypoints_person[part[1], 2] > 0:
                # 布置网格
                x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                out_limb = PointToSegDist(x,y,keypoint_1[0],keypoint_1[1],keypoint_2[0],keypoint_2[1],img)
                out_limb = 1/(2*3.14*sigma_limb*sigma_limb) * np.exp(-out_limb/(2*sigma_limb*sigma_limb))
                #out_limb = out_limb/out_limb.max()
                out_limbSigma[i] = np.where(out_limbSigma[i] < out_limb , out_limb , out_limbSigma[i])

    return out_limbSigma

def PointToSegDist(x,y,x1,y1,x2,y2 , img):
    eps = 1e-7
    out = np.zeros((img.shape[0] , img.shape[1]))
    #求点到线段的距离
    cross = (x2 - x1) * (x - x1) + (y2 - y1) * (y - y1)
    mask1 = cross <= 0
    out = np.where(mask1 ,np.sqrt((x - x1) * (x - x1) + (y - y1) * (y - y1)) , out )
    d2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + eps
    mask2 = cross >= d2
    out = np.where(mask2 ,np.sqrt((x - x2) * (x - x2) + (y - y2) * (y - y2)) , out )
    mask3 =  1- (mask1|mask2)
    r = cross / d2
    px = x1 + (x2 - x1) * r
    py = y1 + (y2 - y1) * r
    out = np.where(mask3 ,np.sqrt((x - px) * (x - px) + (py - y) * (py - y)) , out )
    return out



def add_neck(keypoints):
    right_shoulder = keypoints[6, :]
    left_shoulder = keypoints[5, :]
    neck = np.zeros(3)
    if right_shoulder[2] > 0 and left_shoulder[2] > 0:
        neck = (right_shoulder + left_shoulder) / 2
        neck[2] = 2

    neck = neck.reshape(1, len(neck))
    neck = np.round(neck)
    keypoints = np.vstack((keypoints,neck))

    return keypoints


def get_keypoints_bbox(coco, img, annots):
    keypoints = []
    bbox = []
    for annot in annots:
        #修改keypoint的数据格式，X,Y,V(V表示是否可见，0代表未标注)
        person_keypoints = np.array(annot['keypoints']).reshape(-1, 3)
        #通过左肩和右肩推测neck的位置
        person_keypoints = add_neck(person_keypoints)
        keypoints.append(person_keypoints)
        # 获得bbox数据，（x,y,wx,wy）
        person_bbox = np.array(annot['bbox'])
        bbox.append(person_bbox)
    return np.array(keypoints) , np.array(bbox)


def get_ignore_mask(coco, img, annots):
    #返回图片上需要忽视的位置，False代表这个区域没有东西，True代表这个区域有一些点需要忽略
    mask_union = np.zeros((img.shape[0], img.shape[1]), 'bool')
    masks = []
    for annot in annots:
        mask = coco.annToMask(annot).astype('bool')
        masks.append(mask)
        if check_annot(annot):
            mask_union = mask_union | mask
    ignore_mask = np.zeros((img.shape[0], img.shape[1]), 'bool')
    for i in range(len(annots)):
        annot = annots[i]
        mask = masks[i]
        if not check_annot(annot):
            ignore_mask = ignore_mask | (mask & ~mask_union)

    return ignore_mask.astype('uint8')


def clean_annot(coco, data_path, split):
    #返回包含有效人体的图片index
    ids_path = os.path.join(data_path, split + '_ids.pkl')
    if os.path.exists(ids_path):
        print('Loading filtered annotations for {} from {}'.format(split,ids_path))
        with open(ids_path, 'rb') as f:
            return pickle.load(f)
    else:
        print('Filtering annotations for {}'.format(split))
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
            if i%100==0:
                print(i)
        indices = indices[:valid_count]
        print('Saving filtered annotations for {} to {}'.format(split, ids_path))
        with open(ids_path, 'wb') as f:
            pickle.dump(indices, f)
        return indices

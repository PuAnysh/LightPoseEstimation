from networks import ShuffleNet,MobileNet
import  torch
import cv2
import config as cfg
import  numpy as np
import json
from decode.estimator import ResEstimator
from evaluate.datatransform.DSMAP import *
from evaluate.datatransform.LSPtoAI import MyEncoder
import time
import os

def expand_bbox(left, right, top, bottom, img_width, img_height):
    width = right-left
    height = bottom-top
    ratio = 0.15
    new_left = np.clip(left-ratio*width,0,img_width)
    new_right = np.clip(right+ratio*width,0,img_width)
    new_top = np.clip(top-ratio*height,0,img_height)
    new_bottom = np.clip(bottom+ratio*height,0,img_height)

    return [int(new_left), int(new_top), int(new_right), int(new_bottom)]


def main():
    net_path = cfg.CHECKPOINT_PATH
    IMG_DIR = 'H:\\traindataset\\MPII\\images'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    net = MobileNet(cfg.n_locations)
    e = ResEstimator(net_path, net, cfg.INPUT_SIZE , device)
    anno_file = 'H:\\traindataset\\MPII\\MPII_targetAI.json'
    annos = json.load(open(anno_file, 'r'))
    predice = []
    tolcost = 0
    for anno in annos:
        #MPII数据集中重复的数据需要加上id来区别，有一些不同的valid验证
        img_path = os.path.join(IMG_DIR , anno['image_id'][:13])
        print(img_path)
        image = cv2.imread(img_path)
        x1, y1, x2, y2  = anno['human_annotations']['human1']
        box = expand_bbox(y1,y2,x1,x2,image.shape[0],image.shape[1])
        x1,x2,y1,y2 = box[1],box[3],box[0],box[2]
        # dltx = (x2-x1)//7
        # dlty = (y2-y1)//7
        # x1 = max(x1-dltx , 0)
        # x2 = min(x2+dltx , image.shape[1])
        # y1 = max(y1-dlty,0)
        # y2 = min(y2+dlty,image.shape[0])
        image = image[y1:y2,x1:x2,:]
        img = image.copy()
        pred = {}
        human_ = {}
        with torch.no_grad():
            ts = time.time()
            humans , heat_map = e.inference(img)
            tolcost += time.time() - ts
            image_out = ResEstimator.draw_humans(image, humans, imgcopy=False)
            # cv2.imshow('title' , image_out)
            # cv2.waitKey()
            humans = humans[:, 0:2] + [x1, y1]
            human = []
            for i in range(humans.shape[0]):
                idx = MPII2AI[i]
                if idx == -1:
                    continue
                human.append(humans[idx][0])
                human.append(humans[idx][1])
                human.append(1)
            human_['human1'] = human
            # print(y1,x1)
            # print(human)
            # print(anno['keypoint_annotations']['human1'])

        pred['image_id'] = anno['image_id']
        print(anno['image_id'])
        pred['keypoint_annotations'] = human_
        predice.append(pred)

    jsonfile = json.dumps(predice ,cls=MyEncoder)
    print(tolcost)
    f = open('LSP_prediceAI.json', "w")
    f.write(jsonfile)
    f.close()

if __name__ == '__main__':
    main()





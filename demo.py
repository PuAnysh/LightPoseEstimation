from networks import ShuffleNet,MobileNet
import  torch
import cv2
import config as cfg
import  numpy as np
from decode.estimator import ResEstimator
import time
#net_path = 'H:\\NET\\MobilePose-pytorch\\checkpoint\\mobilenetv2_224_adam_best.t7'
net_path = cfg.CHECKPOINT_PATH
#modelname = 'mobilenetv2'
#img_path = 'H:\\traindataset\\LSP\\images\\im1150.jpg'
img_path = 'H:\\9.png'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
net = MobileNet(cfg.n_locations)
e = ResEstimator(net_path, net, cfg.INPUT_SIZE , device)
image = cv2.imread(img_path)
img = image.copy()
with torch.no_grad():
    ts = time.time()
    for i in range(100):
        humans , heat_map = e.Evalinference(img)
    es = time.time()
    print(1 / ((time.time() - ts)/100))
    #print(humans)
    image_out = ResEstimator.draw_humans(image, humans, imgcopy=False)

    # for i in range(heat_map.shape[1]):
    #     heat = heat_map[0,i,:,:].detach().cpu().numpy()
    #     heat = (heat / heat.max() * 255.).astype('uint8')
    #     colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    #     heat = cv2.resize(colored , (image.shape[1], image.shape[0]))
    #     heat = cv2.addWeighted(img, 0.6, heat, 0.4, 0)
    #     cv2.imshow('MobilePose Demo:{}'.format(i), heat)
    #     cv2.waitKey()
    cv2.imshow('MobilePose Demo', image_out)
    cv2.waitKey()
# #cv2.imshow('title',input[0,:,:,:].transpose((1,2,0)))
#cv2.waitKey()
#print(heatmaps.shape)
#print(coords)
#for i in range(heatmaps.shape[1]):
#    heatmap = heatmaps.detach().cpu().numpy()[0, i, :, :]
#    heatmap = cv2.resize(heatmap, (224, 224))
#    print(heatmap.shape)
#    cv2.imshow('title', heatmap*255)
#    cv2.waitKey()



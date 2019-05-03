n_locations = 16
n_limbs = 17
INPUT_SIZE = 32*12
HEAT_MAP_SIZE = 48
model = 'shufflenetv2'
# training setting
LR = 0.001
BATCH_SIZE = 16
NUM_WORKS = 0
MAX_EPOCH = 400
#CHECKPOINT_PATH = 'C:\\Users\\HuangYC\\Desktop\\毕业论文\\实验数据\\mbnV2BN\\chechpoint_mobilenetv2_CPM_MPII.pth'
CHECKPOINT_PATH = 'H:\\LightNet_Pose_Estimation_build_heatmap\\LightPoseEstimation\\checkpoint\\chechpoint_shufflenetv2_CPM_MPII.pth'
train_log = 'train_log.out'
valid_log = 'valid_log.out'

#evaluate
keypoint_predictions_file = ''
keypoint_annotations_file = ''

# data setting
DATA_DIR = ''
IMG_DIR = 'H:\\traindataset\\MPII'
#IMG_DIR = '/data/zhangjinjin/coco/train2014'
#annotation
keyPoints_ANNOTATION_train_DIR = 'H:\\traindataset\COCO\\annotations\\instances_train2017.json'
Instances_ANNOTATION_train_DIR = 'H:\\traindataset\COCO\\annotations\\person_keypoints_train2017.json'

#keyPoints_ANNOTATION_train_DIR = '/data/zhangjinjin/coco/annotations/instances_train2014.json'
#Instances_ANNOTATION_train_DIR = '/data/zhangjinjin/coco/annotations/person_keypoints_train2014.json'

keyPoints_ANNOTATION_val_DIR = ''
Instances_ANNOTATION_val_DIR = ''

#sigama
sigmaHM = 3
sigma_limb = 3

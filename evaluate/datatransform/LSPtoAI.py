from evaluate.datatransform.DSMAP import *
import numpy as np
import json
from scipy.io import loadmat

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def main():
    data = loadmat('joints.mat')
    import numpy as np
    data = np.array(data['joints'])

    annotations = []
    data = data.astype(np.int32)
    # dumps 将数据转换成字符串
    for i in range(data.shape[2]):
        annotation = {}
        annotation['image_id'] = 'im{:04d}'.format(i)
        points = []
        xmin = data[0, 0, i]
        xmax = data[0, 0, i]
        ymin = data[0, 0, i]
        ymax = data[0, 0, i]
        for ij in range(data.shape[1]):
            j = LSP2AImap[ij]
            xmin = min(data[0, j, i], xmin)
            xmax = max(data[0, j, i], xmax)
            ymin = min(data[1, j, i], ymin)
            ymax = min(data[1, j, i], ymax)
            points.append(data[0, j, i])
            points.append(data[1, j, i])
            points.append(data[2, j, i])
        human = {'human1': points}
        annotation['keypoint_annotations'] = human
        annotation['human_annotations'] = {'human1': [xmin, ymin, xmax, ymax]}
        annotations.append(annotation)

    jsonfile = json.dumps(annotations, cls=MyEncoder)

if __name__ == '__main__':
    main()

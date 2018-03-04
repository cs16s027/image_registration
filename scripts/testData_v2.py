import os
import numpy as np
import cv2
np.random.seed(0)

def getData(images, labels):
    data = {int(i) : [] for i in set(labels)}
    for i in range(images.shape[0]):
        image = images[i]
        label = int(labels[i])
        if len(data[label]) == 1:
            continue
        data[label].append(image)
    return data

def translate(iimage, trange = 25):
    num = 1
    data = []
    for i in range(num):
        image = np.zeros((64, 64), dtype = np.float32)
        x, y = np.random.randint(0, 64-32, 2)
        tvector = np.random.randint(-trange, trange, 2)
        while x + tvector[1] + iimage.shape[0] > 64 or y + tvector[0] + iimage.shape[1] > 64\
        or x + tvector[1] < 0 or y + tvector[0] < 0:
            x, y = np.random.randint(0, 64-32, 2)
            tvector = np.random.randint(-trange, trange, 2)
        image[x : x + iimage.shape[0], y : y + iimage.shape[1]] = iimage
        tmatrix = np.float32([ [1, 0, tvector[0]], [0, 1, tvector[1]] ])
        trans_image = cv2.warpAffine(image, tmatrix, (64, 64))
        data.append((image, trans_image, tvector))
    return data

def writeData(stage, stage_data):
    data = []
    metadata = open('data/%s.txt' % stage, 'w')
    for label in stage_data.keys():
        iimages = stage_data[label]
        iimages_index = len(iimages) - 1
        while iimages_index >= 0:
            iimage = iimages.pop()
            if np.random.random() <= 0.7:
                trange = 25
            else:
                trange = 25
            data = translate(iimage, trange)
            for index, (image, trans_image, tvector) in enumerate(data):
                image_path = 'data/%s/%s_%s_%s_ref.jpg' % (stage, str(label), str(iimages_index + 1), str(index + 1))
                trans_image_path = 'data/%s/%s_%s_%s_def.jpg' % (stage, str(label), str(iimages_index + 1), str(index + 1))
                x, y = tvector
                cv2.imwrite(image_path, image)
                cv2.imwrite(trans_image_path, trans_image)
                metadata.write('%s,%s,%s,%s\n' % (image_path, trans_image_path, str(x), str(y)))
            iimages_index -= 1
    metadata.close()

if __name__ == '__main__':
    base = '/home/iitm/work/image_registration/image_registration/data/devnagari'
    label = 0
    images, labels = [], []
    for f in os.listdir(base):
        path = os.path.join(base, f)
        image = cv2.imread(os.path.join(path, os.listdir(path)[0]), 0)
        images.append(image)
        labels.append(label)
        label += 1
    images = np.array(images)
    data = getData(images, labels)
    writeData('test-devnagari', data)


import numpy as np
import cv2
from itertools import permutations
import h5py
np.random.seed(0)

def getImage(f):
    data = [line.strip().split(',') for line in open(f, 'r').readlines()]
    np.random.shuffle(data)
    for ref_path, def_path, x, y in data:
        # Pick only a particular image
        if '003988' in ref_path:
            return np.float32(cv2.imread(ref_path, 0))

def prepareTextbook():
    ranges = np.arange(-25, 26)
    tvectors = permutations(ranges, 2)
    textbook = {'0,0,3,100' : [], '1,3,6,10' : [], '2,6,11,2' : [], '3,11,15,2' : [], '4,15,20,1' : [], '5,20,26,1' : []}
    for tvector in tvectors:
        l1 = np.abs(np.array(tvector)).sum()
        for topic in textbook.keys():
            epoch, start, end, num = [int(word) for word in topic.split(',')]
            if l1 >= start and l1 < end:
                textbook[topic].append(tvector)
                break
    return textbook

def translate(iimage, tvector, num = 100):
    data = []
    for i in range(num):
        # Initialize images
        image = np.zeros((256, 256), dtype = np.float32)
        trans_image = np.zeros((256, 256), dtype = np.float32)
        # Pick a random point in the input image
        x, y = np.random.randint(0, iimage.shape[0] - image.shape[0], 2)
        # Keep sampling until the image fits
        while x + tvector[1] + image.shape[0] >= iimage.shape[0] or y + tvector[0] + image.shape[1] >= iimage.shape[0]\
        or x + tvector[1] <= 0 or y + tvector[0] <= 0:
            x, y = np.random.randint(0, iimage.shape[0] - image.shape[0], 2)
        # Fixed image
        image[:, :] = iimage[x : x + image.shape[0], y : y + image.shape[1]]
        # Floating image
        x_trans, y_trans = x + tvector[1], y + tvector[0]
        trans_image[:, :] = iimage[x_trans : x_trans + image.shape[0], y_trans : y_trans + image.shape[1]]
        # - for t-vector since this is in already aligned state
        data.append((image, trans_image, -tvector))
    return data

# Generate pairs for this image according to the syllabus
def generate(image, textbook):
    data = {}
    for topic in textbook.keys():
        data[topic] = []
        epoch, start, end, num = topic.split(',')
        for vector in textbook[topic]:
            data[topic] += translate(image, np.array(vector), int(num))
    return data

def writeData(data):
    for topic in data.keys():
        epoch, start, end, num = topic.split(',')
        points = data[topic]
        X, Y = [], []
        for point in points:
            image, trans_image, tvector = point
            X.append(np.stack([image, trans_image], axis = 0))
            Y.append(tvector)
        X = np.array(X, dtype = np.float32)
        Y = np.array(Y, dtype = np.float32)
        print('Epoch : {} has {} images'.format(epoch, Y.shape[0]))
        dataset = h5py.File('data/KLA/train/{}.h5'.format(epoch), 'w')
        dataset.create_dataset(name = 'X', dtype = X.dtype, shape = X.shape, data = X)
        dataset.create_dataset(name = 'Y', dtype = Y.dtype, shape = Y.shape, data = Y)       
        dataset.close()

if __name__ == '__main__':
    image = getImage('/home/iitm/work/image_registration/image_registration/data/KLA/data.txt')
    textbook = prepareTextbook()
    data = generate(image, textbook)
    writeData(data)


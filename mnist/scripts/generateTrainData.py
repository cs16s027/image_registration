from keras.datasets import mnist
import numpy as np
import cv2
import h5py
from itertools import permutations
np.random.seed(0)

def getImages(images, labels):
    data = {i : [] for i in range(10)}
    for i in range(images.shape[0]):
        image = images[i]
        label = int(labels[i])
        if len(data[label]) == 100:
            continue
        data[label].append(image)
    return data

def prepareTextbook():
    ranges = np.arange(-25, 26)
    tvectors = permutations(ranges, 2)
    #textbook_ = {'0,0,1' : [], '1,1,2' : [], '2,2,3'  : [],\
    #        '3,3,4' : [], '4,4,5' : [], '5,5,6'  : [],'6,6,7' : [],\
    #             '7,7,8' : [], '8,8,9' : [], '9,9,10' : []}
    #textbook_ = {'0,0,3' : [], '1,3,6' : [], '2,6,11' : [], '3,11,15' : [], '4,15,20' : [], '5,20,26' : []}
    textbook_ = {'1,0,5' : [], '2,6,10' : [], '3,11,15' : [], '4,16,20' : [] , '5,21,25' : []}
    for tvector in tvectors:
        l1 = np.abs(np.array(tvector)).sum()
        for topic in textbook_.keys():
            epoch, start, end = [int(word) for word in topic.split(',')]
            if l1 >= start and l1 < end:
                textbook_[topic].append(tvector)
                break
    textbook_['1,0,5'].append((0, 0))
    textbook = {}
    for topic_ in textbook_.keys():
        epoch, start, end = topic_.split(',')
        topic = ','.join([epoch, start, end, '1'])
        textbook[topic] = textbook_[topic_]
    return textbook

def translate(iimage, tvector, num = 100):
    data = []
    for i in range(num):
        image = np.zeros((64, 64), dtype = np.float32)
        x, y = np.random.randint(0, 64-28, 2)
        while x + tvector[1] + iimage.shape[0] > 64 or y + tvector[0] + iimage.shape[1] > 64\
        or x + tvector[1] < 0 or y + tvector[0] < 0:
            x, y = np.random.randint(0, 64-28, 2)
        image[x : x + iimage.shape[0], y : y + iimage.shape[1]] = iimage
        tmatrix = np.float32([ [1, 0, tvector[0]], [0, 1, tvector[1]] ])
        trans_image = cv2.warpAffine(image, tmatrix, (64, 64))
        data.append((image, trans_image, tvector))
    return data

# Generate pairs for this image according to the syllabus
def generate(image, textbook, data):
    for topic in textbook.keys():
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
        x = np.arange(X.shape[0])
        np.random.shuffle(x)
        X, Y = X[x[ : 2000]], Y[x[ : 2000]]
        print('Epoch : {} has {} images'.format(epoch, Y.shape[0]))
        dataset = h5py.File('data/train/{}.h5'.format(epoch), 'w')
        dataset.create_dataset(name = 'X', dtype = X.dtype, shape = X.shape, data = X)
        dataset.create_dataset(name = 'Y', dtype = Y.dtype, shape = Y.shape, data = Y)       
        dataset.close()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    images = getImages(x_train, y_train)
    textbook = prepareTextbook()
    data = {}
    for topic in textbook.keys():
        data[topic] = []
    for label in images.keys():
        for image in images[label]:
            data = generate(image, textbook, data)
    writeData(data)


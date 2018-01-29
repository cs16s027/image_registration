import numpy as np
import cv2
import h5py

for stage in ['train', 'test']:
    data = [line.strip().split(',') for line in open('data/KLA/%s.txt' % stage, 'r').readlines()]
    np.random.shuffle(data)
    X, Y = [], []
    for image_path, trans_image_path, x, y in data:
        image = cv2.imread(image_path, 0)
        trans_image = cv2.imread(trans_image_path, 0)
        tvector = np.float32([x, y])
        X.append(np.stack([image, trans_image], axis = 0))
        Y.append(tvector)
    X = np.array(X, dtype = np.float32)
    Y = np.float32(Y, dtype = np.float32)
    dataset = h5py.File('data/KLA/%s.h5' % stage, 'w')
    dataset.create_dataset(name = 'X', dtype = X.dtype, shape = X.shape, data = X)
    dataset.create_dataset(name = 'Y', dtype = Y.dtype, shape = Y.shape, data = Y)

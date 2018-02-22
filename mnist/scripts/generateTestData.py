from keras.datasets import mnist
import numpy as np
import cv2
import h5py
np.random.seed(1234)

def getData(images, labels):
    data = {i : [] for i in range(10)}
    for i in range(images.shape[0]):
        image = images[i]
        label = int(labels[i])
        if len(data[label]) == 100:
            continue
        data[label].append(image)
    return data

def translate(digit, trange = 25):
    num = 10
    data = []
    for i in range(num):
        image = np.zeros((64, 64), dtype = np.float32)
        x, y = np.random.randint(0, 64-28, 2)
        tvector = np.random.randint(-trange, trange, 2)
        while x + tvector[1] + digit.shape[0] > 64 or y + tvector[0] + digit.shape[1] > 64\
        or x + tvector[1] < 0 or y + tvector[0] < 0:
            x, y = np.random.randint(0, 64-28, 2)
            tvector = np.random.randint(-trange, trange, 2)
        image[x : x + digit.shape[0], y : y + digit.shape[1]] = digit
        tmatrix = np.float32([ [1, 0, tvector[0]], [0, 1, tvector[1]] ])
        trans_image = cv2.warpAffine(image, tmatrix, (64, 64))
        data.append((image, trans_image, tvector))
    return data

def writeData(data, dataset_name):
    X, Y = [], []
    for label in data.keys():
        digits = data[label]
        digits_index = len(digits) - 1
        while digits_index >= 0:
            digit = digits.pop()
            data_ = translate(digit, trange = 10)
            for index, (image, trans_image, tvector) in enumerate(data_):
                X.append(np.stack([image, trans_image], axis = 0))
                Y.append(tvector)
            digits_index -= 1
    X = np.array(X, dtype = np.float32)
    Y = np.array(Y, dtype = np.float32)
    x = np.arange(X.shape[0])
    np.random.shuffle(x)
    X, Y = X[x], Y[x]

    dataset = h5py.File(dataset_name, 'w')
    dataset.create_dataset(name = 'X', dtype = X.dtype, shape = X.shape, data = X)
    dataset.create_dataset(name = 'Y', dtype = Y.dtype, shape = Y.shape, data = Y)       
    dataset.close()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    test = getData(x_test, y_test)
    writeData(test, 'data/test/MNIST.h5')


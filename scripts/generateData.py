from keras.datasets import mnist
import numpy as np
import cv2
np.random.seed(0)

def getData(images, labels):
    data = {i : [] for i in range(10)}
    for i in range(images.shape[0]):
        image = images[i]
        label = int(labels[i])
        if len(data[label]) == 100:
            continue
        data[label].append(image)
    return data

def translate(digit):
    num = 10
    trange = 5
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

def writeData(stage, stage_data):
    data = []
    metadata = open('data/%s.txt' % stage, 'w')
    for label in stage_data.keys():
        digits = stage_data[label]
        digits_index = len(digits) - 1
        while digits_index >= 0:
            digit = digits.pop()
            data = translate(digit)
            for index, (image, trans_image, tvector) in enumerate(data):
                image_path = 'data/%s/%s_%s_%s_ref.jpg' % (stage, str(label), str(digits_index + 1), str(index + 1))
                trans_image_path = 'data/%s/%s_%s_%s_def.jpg' % (stage, str(label), str(digits_index + 1), str(index + 1))
                x, y = tvector
                cv2.imwrite(image_path, image)
                cv2.imwrite(trans_image_path, trans_image)
                metadata.write('%s,%s,%s,%s\n' % (image_path, trans_image_path, str(x), str(y)))
            digits_index -= 1
    metadata.close()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    train = getData(x_train, y_train)
    test  = getData(x_test, y_test)
    writeData('train', train)
    writeData('test', test)


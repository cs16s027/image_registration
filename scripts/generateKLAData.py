import numpy as np
import cv2
np.random.seed(0)

def getData(f):
    data = [line.strip().split(',') for line in open(f, 'r').readlines()]
    np.random.shuffle(data)
    images = []
    for ref_path, def_path, x, y in data:
        ref_image = cv2.imread(ref_path, 0)
        assert ref_image.shape[0] == 512 and ref_image.shape[1] == 512
        images.append(ref_image)
    images = np.float32(images)
    return images

def translate(iimage, trange = 25):
    num = 20
    data = []
    for i in range(num):
        image = np.zeros((256, 256), dtype = np.float32)
        trans_image = np.zeros((256, 256), dtype = np.float32)
        x, y = np.random.randint(0, iimage.shape[0] - image.shape[0], 2)
        tvector = np.random.randint(-trange, trange, 2)
        while x + tvector[1] + image.shape[0] >= iimage.shape[0] or y + tvector[0] + image.shape[1] >= iimage.shape[0]\
        or x + tvector[1] <= 0 or y + tvector[0] <= 0:
            x, y = np.random.randint(0, iimage.shape[0] - image.shape[0], 2)
            tvector = np.random.randint(-trange, trange, 2)
        image[:, :] = iimage[x : x + image.shape[0], y : y + image.shape[1]]
        x_trans, y_trans = x + tvector[1], y + tvector[0]
        trans_image[:, :] = iimage[x_trans : x_trans + image.shape[0], y_trans : y_trans + image.shape[1]]
        data.append((image, trans_image, tvector))
    return data

def generate(images):
    data = [] 
    for image in images:
        data += translate(image)
    np.random.shuffle(data)
    return data

def partition(data):
    return data[ : int(0.7 * len(data))], data[int(0.7 * len(data)) : ]

def writeData(stage, images):
    metadata = open('data/KLA/%s.txt' % stage, 'w')
    for index, (image, trans_image, tvector) in enumerate(images):
        image_path = 'data/KLA/%s/%s_ref.jpg' % (stage, str(index + 1))
        trans_image_path = 'data/KLA/%s/%s_def.jpg' % (stage, str(index + 1))
        x, y = tvector
        cv2.imwrite(image_path, image)
        cv2.imwrite(trans_image_path, trans_image)
        metadata.write('%s,%s,%s,%s\n' % (image_path, trans_image_path, str(x), str(y)))
    metadata.close()

if __name__ == '__main__':
    images = getData('/home/iitm/work/image_registration/image_registration/data/KLA/data.txt')
    data = generate(images)
    train, test = partition(data)
    writeData('train', train)
    writeData('test', test)


import cv2
from numpy import *
import os
import numpy as np

# Path to training images for 10 objects
path = {}
query_path = []
objects = ['bear', 'butterfly', 'coffee-mug', 'elk', 'fire-truck', 'horse', 'hot-air-balloon', 'iris', 'owl', 'teapot']
for i in range(len(objects)):
    path[i] = 'dataset/train/%s/' % (objects[i])

    # path to test images
    query_path = "dataset/query/"

# parameters
first_index = 0  # starting index of training images
last_index = 29  # last index of training images
total = last_index - first_index  # number of images
gabor_filters = 40  # total gabor filters
ksize = 7   # gabor filter kernel size
distype = "euclidean"  # or "cosine"
select = "gabor"  # or "sift"


# lambd_no = 4

def build_filters(gabor_filters):
    filters = []
    # define the range for theta and lambd
    for theta in np.arange(0, np.pi, np.pi / gabor_filters):
        # for lambd in np.arange(0, 6*np.pi/4, np.pi / lambd_no):for changing lambda write below lambda_no instead of 10
        kern = cv2.getGaborKernel((ksize, ksize), 1.0, theta, 10, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def feature(Path, I1, I2):
    filters = build_filters(gabor_filters)
    f = np.asarray(filters)
    feature = []

    listing = sorted(os.listdir(Path))
    for image in listing[I1:I2]:
        im = cv2.imread(Path + image)
        temp = 0
        for j in range(gabor_filters):
            res = process(im, f[j])
            temp = temp + res
        # calculating the mean amplitude for each convolved image
        feature.append(np.mean(temp))
    return feature


from scipy import spatial


def distance(test, img, distype):

    if distype == "euclidean":
        return np.linalg.norm(test - img)

    elif distype == "cosine":
        c = spatial.distance.cosine(test, img)
        return 1 - c


def knn(test, train, distype):
    minx = float('inf')
    min_id = 0
    for k in range(len(train)):
        img = train[k]
        dis = distance(test, img, distype)
        if dis <= minx:
            minx = dis
            min_id = k
    return min_id


def average_sift(image):
    sift = cv2.xfeatures2d.SIFT_create()
    kp, des = sift.detectAndCompute(image, None)
    avgdes = np.zeros((1, 128), dtype=np.float)
    for i in range(128):
        avgdes[0, i] = np.average(des[:, i])
    return avgdes


def sift_feature(Path, I1, I2):
    listing = sorted(os.listdir(Path))
    feature = []
    for image in listing[I1:I2]:
        img = cv2.imread(Path + image)
        res = average_sift(img)
        feature.append(res)
    return feature


def main(select):

    treainfeature = {}

    # train feature
    for x in range(len(objects)):
        if select == "gabor":
            treainfeature[x] = feature(path[x], first_index, last_index+1)
        elif select == "sift":
            treainfeature[x] = sift_feature(path[x], first_index, last_index+1)

    testfeature = []
    # test feature
    if select == "gabor":
        testfeature = feature(query_path, 0, 50)  # 50 = query image size
    elif select == "sift":
        testfeature = sift_feature(query_path, 0, 50)  # 50 = query image size

    testfeature = np.array(testfeature)
    testfeature.astype(float32)

    results = []
    for j in range(50):
        temp = knn(testfeature[j], treainfeature, distype)
        results.append(temp)

    groundtruth = np.array([[0], [0], [0], [0], [0],
                            [1], [1], [1], [1], [1],
                            [2], [2], [2], [2], [2],
                            [3], [3], [3], [3], [3],
                            [4], [4], [4], [4], [4],
                            [5], [5], [5], [5], [5],
                            [6], [6], [6], [6], [6],
                            [7], [7], [7], [7], [7],
                            [8], [8], [8], [8], [8],
                            [9], [9], [9], [9], [9]])
    count = 0


    for a in range(len(groundtruth)):
        if results[a] == groundtruth[a]:
            count = count + 1


    print("general accuracy: " + str(count * 100 / len(groundtruth)))

    from sklearn.metrics import accuracy_score

    list = []
    print("class based accuracy: ")
    for a in range(len(objects)):
        cm = accuracy_score(groundtruth[(a*5):((a+1)*5)], results[(a*5):((a+1)*5)])
        list.append(objects[a]+": "+str(cm))
    print('[%s]' % ', '.join(map(str, list)))



'''		
    listing = sorted(os.listdir(query_path))
    print("true predicted images:")
    for i in list:
        print(listing[i])
'''


if __name__ == '__main__':
    main(select)

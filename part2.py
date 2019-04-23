from numpy import *
import numpy as np
import cv2
import os

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
clusters = 25  # number of clusters in k-means
distype = "euclidean"  # or "cosine"
tile = "no"  # or "yes"


import image_slicer


# Function to extract sift features from images of given class and concatenate them into single matrix
def sift_descriptor(Path, I1, I2):
    des_train = None
    size = None
    listing = sorted(os.listdir(Path))
    for image in listing[I1:I2]:
        pat = Path + image
        if tile == "yes":
            im = image_slicer.slice(pat, 4)
        else:
            im = cv2.imread(pat)
        # finding SIFT key points and descriptors
        sift = cv2.xfeatures2d.SIFT_create()
        (kp_train, des1_train) = sift.detectAndCompute(im, None)
        size1 = np.shape(des1_train)
        if des_train is None:
            des_train = des1_train
            size = np.array([size1])
        else:
            des_train = np.concatenate((des_train, des1_train), axis=0)
            size = np.concatenate((size, np.array([size1])), axis=0)

    return des_train, size[:, 0]


# Function to create histogram representation of images after K-means clustering over all the SIFT features of all classes
def create_hist(codebooks, clusters, size_class, sum_features, classidx):
    idx = np.concatenate((np.array([0]), size_class))
    range1 = np.sum(sum_features[0:classidx])
    range2 = range1 + sum_features[classidx]
    codebooks_temp = codebooks[range1:range2]
    R1 = np.shape(idx)
    hist = None
    for i in range(0, R1[0] - 1):
        range1 = np.sum(idx[0:i + 1])
        range2 = range1 + idx[i + 1]
        hist_temp, bin_edges = np.histogram(codebooks_temp[range1:range2], clusters)
        if hist is None:
            hist = hist_temp
        else:
            hist = np.vstack((hist, hist_temp))
    return hist


# Function to encode the given test image into histogram feature
def create_testfeature(Path, I3, centers, clusters, distype):
    listing = sorted(os.listdir(Path))
    test_img = cv2.imread(Path + listing[I3])
    sift = cv2.xfeatures2d.SIFT_create()
    (kp_train, dest1_train) = sift.detectAndCompute(test_img, None)
    no_descr = np.shape(dest1_train)
    dist = np.zeros((no_descr[0], clusters))
    Min = np.zeros(no_descr[0])
    for i in range(0, no_descr[0]):
        for j in range(0, clusters):
            a = dest1_train[i, :]
            b = centers[j, :]
            if distype == "euclidean":
                #  Euclidean distance
                dist[i, j] = np.linalg.norm(a - b)
            elif distype == "cosine":
                #  Cosine similarity
                dot = np.dot(a, b)
                norma = np.linalg.norm(a)
                normb = np.linalg.norm(b)
                dist[i, j] = dot / (norma * normb)
        Min[i] = np.argmin(dist[i, :])
    Min1 = Min.astype(np.float32)
    hist_test, bin_edges = np.histogram(Min1, clusters)
    return hist_test.astype(float32)

def distance(test, img, distype):

    if distype == "euclidean":
        return np.linalg.norm(test - img)

    elif distype == "cosine":
        dot = np.dot(test, img)
        norma = np.linalg.norm(test)
        normb = np.linalg.norm(img)
        return dot / (norma * normb)


def knn(test, train, distype):  # euclidean or cosine similarity
    min = float('inf')
    min_id = 0
    for i in range(len(train)):
        img = train[i]
        dis = distance(test, img, distype)
        if dis <= min:
            min = dis
            min_id = i
    return min_id


if __name__ == '__main__':
    descr_objects = {}
    object_size = {}
    addi = {}

    # Extract SIFT Features of images from all class and concatenate into same Matrix
    for x in range(len(objects)):
        descr_objects[x], object_size[x] = sift_descriptor(path[x], first_index, last_index)
        addi[x] = np.sum(object_size[x])

    descriptors = np.concatenate((descr_objects[0], descr_objects[1], descr_objects[2], descr_objects[3], descr_objects[4],
                                  descr_objects[5], descr_objects[6], descr_objects[7], descr_objects[8], descr_objects[9]), axis=0)
    sum_features = np.hstack(([0], addi[0], addi[1], addi[2], addi[3], addi[5], addi[6], addi[7], addi[8], addi[9]))

    # performing k-means clustering on the descriptors

    # Define criteria = ( type, max_iter = 20 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    # Set flags
    flags = cv2.KMEANS_RANDOM_CENTERS
    compactness, codebooks, centers = cv2.kmeans(descriptors, clusters, None, criteria, 20, flags)

    # create histogram of codebooks for every objects
    Hist_objects = {}
    for x in range(len(objects)):
        Hist_objects[x] = create_hist(codebooks, clusters, object_size[x], sum_features, x)

    # query histogram
    test_list = sorted(os.listdir(query_path))
    size = np.shape(test_list)
    hist_query = None
    for i in range(0, size[0]):
        # euclidean distance or cosine similarity selected here
        Hist_test1 = create_testfeature(query_path, i, centers, clusters, distype)
        if hist_query is None:
            hist_query = Hist_test1
        else:
            hist_query = np.vstack((hist_query, Hist_test1))

    results = []
    for i in range(50):
        temp = knn(hist_query[i], Hist_objects, distype)
        results.append(temp)

    groundtruth = np.array([[0],[0],[0],[0],[0],
                          [1],[1],[1],[1],[1],
                          [2],[2],[2],[2],[2],
                          [3],[3],[3],[3],[3],
                          [4],[4],[4],[4],[4],
                          [5],[5],[5],[5],[5],
                          [6],[6],[6],[6],[6],
                          [7],[7],[7],[7],[7],
                          [8],[8],[8],[8],[8],
                          [9],[9],[9],[9],[9]])
    count = 0
    list = []
    for x in range(len(groundtruth)):
        if results[x] == groundtruth[x]:
            count = count + 1
            list.append(x)

    # Result
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




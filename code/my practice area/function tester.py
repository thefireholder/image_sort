import cv2
import numpy
import timeit
from sklearn import neighbors, svm, cluster

def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
    r_image = cv2.resize(input_image,(target_size,target_size)) #resized
    # to be zero-mean, and in the [-1, 1] range.
    output_image = cv2.normalize(r_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return output_image

# imresize test
"""
image = cv2.imread('image2.png',cv2.IMREAD_GRAYSCALE)
o_image = imresize(image,30)
print(o_image)
"""

def reportAccuracy(true_labels, predicted_labels, label_dict):
    # generates and returns the accuracy of a model
    # true_labels is a n x 1 cell array, where each entry is an integer
    # and n is the size of the testing set.
    # predicted_labels is a n x 1 cell array, where each entry is an
    # integer, and n is the size of the testing set. these labels
    # were produced by your system
    # label_dict is a 15x1 cell array where each entry is a string
    # containing the name of that category
    # accuracy is a scalar, defined in the spec (in %)
    accuracy = (true_labels-predicted_labels==0).sum()/true_labels.size
    
    return accuracy

# reportAccuracy test
"""
a=numpy.asarray([1,2,1,4,5,6])
b=numpy.asarray([1,1,1,6,5,3])
print(reportAccuracy(a,b, None))
"""

def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images
    
    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation. You can assume M = N unless you've modified the
    # starter code.
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    KNN = neighbors.KNeighborsClassifier(n_neighbors = num_neighbors)
    KNN.fit(train_features, train_labels)
    predicted_categories = KNN.predict(test_features)
    
    return predicted_categories

# test KNN_classifiers
"""
from sklearn import datasets
from matplotlib import pyplot as plt

#generate 5 2D blobs with X as points, Y as blob it belongs to
blobs_X, blobs_y = datasets.make_blobs(centers=5,random_state=0)
predicted_labels = KNN_classifier(blobs_X, blobs_y, blobs_X, 5)
plt.scatter(blobs_X[:,0], blobs_X[:,1], c=predicted_labels) #plot points
plt.show() #show plot
"""

def tinyImages(train_features, test_features, train_labels, test_labels, label_dict):
    # train_features is a nx1 array of images
    # test_features is a nx1 array of images
    # train_labels is a nx1 array of integers, containing the label values
    # test_labels is a nx1 array of integers, containing the label values
    # label_dict is a 15x1 array of strings, containing the names of the labels
    # classResult is a 18x1 array, containing accuracies and runtimes
    
    accuracies = []
    runtimes = []
    
    for image_scale in [8,16,32]:
        for K in [1,3,6]:
            start = timeit.default_timer()
            
            X_train = numpy.array([imresize(image, image_scale).flatten() for image in train_features])
            y_train = train_labels
            X_test = numpy.array([imresize(image, image_scale).flatten() for image in test_features])
            
            predicted_labels = KNN_classifier(X_train, y_train, X_test, K)
            accuracy = reportAccuracy(test_labels, predicted_labels, label_dict)
            
            end = timeit.default_timer()
            
            accuracies.append(accuracy)
            runtimes.append(end-start)

    classResult = numpy.append(accuracies,runtimes)
    return classResult


"""
#retrieve test and train data
import os
X_train = []
y_train = []
rootDir = '../../data/train' # (NOTE CANNOT HAVE .DS_store in Test folder and Train folder)
n = 0
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName[17:])
    n = n + 1
    for fname in fileList:
        image = cv2.imread('{}/{}'.format(dirName,fname),cv2.IMREAD_GRAYSCALE)
        X_train.append(image)
        y_train.append(n)
        #y_train.append(dirName[17:])

X_train = numpy.asarray(X_train)
y_train = numpy.asarray(y_train)

X_test = []
y_test = []
rootDir = '../../data/test' # (NOTE CANNOT HAVE .DS_store in Test folder and Train folder)
n = 0
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName[16:])
    n = n + 1
    for fname in fileList:
        image = cv2.imread('{}/{}'.format(dirName,fname),cv2.IMREAD_GRAYSCALE)
        X_test.append(image)
        y_test.append(n)
        #y_test.append(dirName[16:])

X_test = numpy.asarray(X_test)
y_test = numpy.asarray(y_test)


# test tinyImages
x = tinyImages(X_train, X_test, y_train, y_test, None)
print(x)

"""
def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.
    
    # train_images is a n x 1 array of images
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"
    
    # the output 'vocabulary' should be dict_size x d, where d is the
    # dimention of the feature. each row is a cluster centroid / visual word.
    
    descriptors = numpy.empty((0,128), int) #get all descriptors
    for image in train_images:
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(image,None)
        descriptors = numpy.append(descriptors,des,axis=0)
    
    print("kmean begin")
    kmeans = cluster.KMeans(dict_size)
    kmeans.fit(descriptors)

    print("kmean ended")
    vocabulary = kmeans.cluster_centers_
    return vocabulary

import os
X_train = []
y_train = []
rootDir = '../../data/train' # (NOTE CANNOT HAVE .DS_store in Test folder and Train folder)
n = 0
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName[17:])
    n = n + 1
    for fname in fileList:
        image = cv2.imread('{}/{}'.format(dirName,fname),cv2.IMREAD_GRAYSCALE)
        X_train.append(image)
        y_train.append(n)
#y_train.append(dirName[17:])

X_train = numpy.asarray(X_train)
y_train = numpy.asarray(y_train)

z = buildDict(X_train[:10],50,"sift","kmeans")
print(z.shape)


"""
img = cv2.imread('image2.png',cv2.IMREAD_GRAYSCALE)
sift = cv2.xfeatures2d.SIFT_create()
kp, des = sift.detectAndCompute(img,None)
print(des[0].shape)
print(len(kp))
"""
#this creates keypoint annotated img
#kp = sift.detect(img,None)
#img=cv2.drawKeypoints(img,kp,img,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
#cv2.imwrite('sift_keypoints.png',img)



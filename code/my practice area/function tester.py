import cv2
import numpy as np
import timeit
from sklearn import neighbors, svm, cluster, metrics


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
    accuracy = metrics.accuracy_score(true_labels, predicted_labels) * 100
    #accuracy = (true_labels-predicted_labels==0).sum()/true_labels.size
    return accuracy

# reportAccuracy test
"""
a=[1,2,1,4,5,6]
b=[1,1,1,6,5,3]
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
            
            X_train = np.array([imresize(image, image_scale).flatten() for image in train_features])
            y_train = train_labels
            X_test = np.array([imresize(image, image_scale).flatten() for image in test_features])
            
            predicted_labels = KNN_classifier(X_train, y_train, X_test, K)
            accuracy = reportAccuracy(test_labels, predicted_labels, label_dict)
            
            end = timeit.default_timer()
            
            accuracies.append(accuracy)
            runtimes.append(end-start)

    classResult = np.append(accuracies,runtimes)
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

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

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

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)


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
    
    descriptors = None;
    d = None;
    xfeature = None;
    
    #choose feature describer
    if feature_type == 'sift':
        d = 128
        xfeature = cv2.xfeatures2d.SIFT_create()
    elif feature_type == 'surf':
        d = 64
        xfeature = cv2.xfeatures2d.SURF_create()
    elif feature_type == 'orb':
        d = 32
        xfeature = cv2.ORB_create(nfeatures=500)

    # get feature
    descriptors = np.empty((0,d), int) #get all descriptors
    for image in train_images:
        kp, des = xfeature.detectAndCompute(image,None)
        
        #down sample key points for each image!
        if type(des) == type(None):
            continue
        
        if len(des) > 50:
            np.random.shuffle(des)
            des = des[0:50] #arbitary keypoint number I chose 500
            descriptors = np.append(descriptors,des,axis=0)

    # get cluster
    vocabulary = None
    if clustering_type == 'kmeans':
        kmeans = cluster.KMeans(dict_size)
        kmeans.fit(descriptors)
        vocabulary = kmeans.cluster_centers_

    if clustering_type == 'hierarchical':
        hierarchical = cluster.AgglomerativeClustering(dict_size)
        hierarchical.fit(descriptors)
        # finding cluster using mean = Sum / N
        c = [[0,np.zeros(d)] for i in range(dict_size)]
        index = 0
        for i in hierarchical.labels_:
            c[i][1] += descriptors[index]
            c[i][0] += 1
            index += 1
        
        vocabulary = np.array([c[i][1] / c[i][0] for i in range(dict_size)])


    return vocabulary

# test build dictionary
"""
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

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)

print("building dictionary")
z = buildDict(X_train[0:10],50,"sift","hierarchical")
print(z.shape)
z = buildDict(X_train[0:10],50,"surf","hierarchical")
print(z.shape)
z = buildDict(X_train[0:10],50,"orb","hierarchical")
print(z.shape)
z = buildDict(X_train[0:10],50,"sift","kmeans")
print(z.shape)
z = buildDict(X_train[0:10],50,"surf","kmeans")
print(z.shape)
z = buildDict(X_train[0:10],50,"orb","kmeans")
print(z.shape)
"""

"""
# sift - small example
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


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary
    
    # BOW is the new image representation, a normalized histogram
    
    xfeature = None
    dict_size = len(vocabulary)
    
    #choose feature describer
    if feature_type == 'sift':
        xfeature = cv2.xfeatures2d.SIFT_create()
    elif feature_type == 'surf':
        xfeature = cv2.xfeatures2d.SURF_create()
    elif feature_type == 'orb':
        xfeature = cv2.ORB_create(nfeatures=500)
    
    # get feature & Bag of word it
    kp, des = xfeature.detectAndCompute(image,None)

    # sometimes des returns None, in that case, return histogram of 0s
    if type(des) == type(None):
        return np.zeros(dict_size)

    """
    #down sample key points for each image!
    if len(des) > 20:
        np.random.shuffle(des)
        des = des[0:20] #arbitary keypoint number I chose 500
    """
        
    # label features
    labeled_des = KNN_classifier(vocabulary, range(dict_size), des, num_neighbors = 1)

    
    #histogram representation
    Bow = np.histogram(labeled_des, bins=range(dict_size + 1))[0]
    return Bow



#testing computeBow on some image

#first we need X training
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

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)


# first we need X test data
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

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

#here we start grabbing Bow and training


for detector in ['sift','surf','orb']:
    for clutser_type in ["kmeans", "hierarchical"]:
        for dict_size in [20, 50]:
            print("buildDict(" + str(dict_size) + ", " + detector + ", " + clutser_type + ")")
            z = buildDict(X_train[:10],dict_size,detector,clutser_type)

print("begin Bow representation for X_train")
X_train_Bow = [computeBow(image,z,"orb") for image in X_train]
print("begin Bow representation for X_test")
X_test_Bow = [computeBow(image,z,"orb") for image in X_test]
print("begin KNN")

predicted_labels = KNN_classifier(X_train_Bow, y_train, X_test_Bow, num_neighbors = 9)
accuracy = reportAccuracy(y_test, predicted_labels, None)

print(accuracy)


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.
    
    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs
    
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    
    clfs = []
    #probabilities = np.empty((0,len(test_features)), int)
    kernel = ''
    
    
    if is_linear:
        kernel = 'rbf'
    else:
        kernel = 'linear'
    """
    for i in range(15):
        clf = svm.SVC(random_state=0,gamma='auto',probability=True,C = svm_lambda,kernel=kernel)
        y = np.where(train_labels == i, 99, train_labels) # making it binary classification
        y = np.where(y != 99, 0, y)
        clf.fit(train_features, y)
        probabilities = np.append(probabilities,[clf.predict_proba(test_features)[:,1]],axis=0)
        print(clf.predict_proba(test_features))
        print(clf.predict(test_features))
        #print(y)
    
    
    predicted_categories = np.argmax(probabilities, axis=0)
    """
    
    
    probabilities = np.zeros((2,len(test_features)))
    for i in range(15):
        y = np.where(train_labels == i, 99, train_labels) # making it binary classification
        y = np.where(y != 99, 0, y)
        clf = svm.SVC(random_state=0,gamma='auto',probability=True,C = svm_lambda,kernel=kernel)
        clf.fit(train_features, y)
        probability=clf.predict_proba(test_features)[:,1]
        for j,x in enumerate(probability):
            if x > probabilities[1][j]:
                probabilities[1][j] = x
                probabilities[0][j] = i
        print(y)
        print(probability)
        print(probabilities[0])
        print("")
        print("")
        print("")
    

    predicted_categories = probabilities[0]
    return predicted_categories

#testing svm concept
"""
clf = svm.SVC(random_state=0,gamma='auto',probability=True)

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

clf.fit(X, y)
print(clf.predict([[1, 1]]))
print(clf.predict_proba([[-1, -1]]))
"""

"""
#testing SVM_classifier on some image
#first we need X training
import os
X_train = []
y_train = []
rootDir = '../../data/train' # (NOTE CANNOT HAVE .DS_store in Test folder and Train folder)
n = -2
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName[17:])
    n = n + 1
    for fname in fileList:
        image = cv2.imread('{}/{}'.format(dirName,fname),cv2.IMREAD_GRAYSCALE)
        X_train.append(image)
        y_train.append(n)
#y_train.append(dirName[17:])

X_train = np.asarray(X_train)
y_train = np.asarray(y_train)
                             
# first we need X test data
X_test = []
y_test = []
rootDir = '../../data/test' # (NOTE CANNOT HAVE .DS_store in Test folder and Train folder)
n = -2
for dirName, subdirList, fileList in os.walk(rootDir):
    #print('Found directory: %s' % dirName[16:])
    n = n + 1
    for fname in fileList:
        image = cv2.imread('{}/{}'.format(dirName,fname),cv2.IMREAD_GRAYSCALE)
        X_test.append(image)
        y_test.append(n)

X_test = np.asarray(X_test)
y_test = np.asarray(y_test)

import sys
np.set_printoptions(threshold=sys.maxsize)
print("Begin building dictionary")
z = buildDict(X_train[:10],20,"sift","kmeans")

print("begin Bow representation for X_train")
X_train_Bow = [computeBow(image,z,"sift") for image in X_train]
print("begin Bow representation for X_test")
X_test_Bow = [computeBow(image,z,"sift") for image in X_test]
print("begin KNN")



y = np.where(y_train == 0, -1, y_train) # making it binary classification
y = np.where(y != -1, 0, y)
y = np.where(y == -1, 1, y)
print(y)
clf = svm.SVC(gamma='auto',probability=True,C = 1)
clf.fit(X_train_Bow, y)
prediction = clf.predict(X_test_Bow)
probability = clf.predict_proba(X_test_Bow)
print(prediction)
print(probability)

"""

"""
prediction = clf.predict(X_train_Bow)
probability = clf.predict_proba(X_train_Bow)
print(prediction)
print(probability)
"""

"""
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 1)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
"""

"""
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 100)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 10)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
#print(predicted_labels)
#print(y_test)
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 1)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 0.1)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 0.01)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
predicted_labels = SVM_classifier(X_train_Bow, y_train, X_test_Bow, False, 0.001)
accuracy = reportAccuracy(y_test, predicted_labels, None)
print(accuracy)
"""

"""
clf = svm.SVC(random_state=0,gamma='auto',probability=True,C = 1,kernel="rbf")
y =np.where(y_train[:300] == 1, 99, y_train[:300])
y =np.where(y != 99, 0, y)

clf.fit(X_train_Bow, y)
probabilities = clf.predict_proba(X_train_Bow)
print(probabilities)
print(clf.predict(X_train_Bow))
print(probabilities[:,0])
print(probabilities.shape)

probabilities = np.empty((0,len(X_train_Bow)), int)
a=clf.predict_proba(X_train_Bow)[:,0]
print(probabilities.shape)
print(np.array([a]).shape)
probabilities = np.append(probabilities,[a],axis=0)
probabilities = np.append(probabilities,[a],axis=0)
print(probabilities)
print(probabilities.shape)


predicted_categories = np.argmax(probabilities, axis=0)
print(predicted_categories.shape)
print(predicted_categories)

"""

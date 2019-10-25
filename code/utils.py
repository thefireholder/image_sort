import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing, metrics


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)

    return train_images, test_images, train_labels, test_labels

#DONE
def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    KNN = neighbors.KNeighborsClassifier(n_neighbors = num_neighbors)
    KNN.fit(train_features, train_labels)
    predicted_categories = KNN.predict(test_features)
    
    return predicted_categories

#DONE
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
    probabilities = np.empty((0,len(test_features)), int)
    kernel = ''
    
    if is_linear:
        kernel = 'rbf'
    else:
        kernel = 'linear'
    
    for i in range(15):
        clf = svm.SVC(random_state=0,gamma='auto',probability=True,C = svm_lambda,kernel=kernel)
        y = np.where(train_labels == i, 99, train_labels) # making it binary classification
        y = np.where(y != 99, 0, y)
        clf.fit(train_features, y)
        probabilities = np.append(probabilities,[clf.predict_proba(test_features)[:,0]],axis=0)
    
    predicted_categories = np.argmax(probabilities, axis=0)
    

    return predicted_categories

#DONE
def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size].
    r_image = cv2.resize(input_image,(target_size,target_size)) #resized
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    output_image = cv2.normalize(r_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return output_image

#DONE
def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    accuracy = metrics.accuracy_score(true_labels, predicted_labels) * 100
    return accuracy

#DONE
def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.
    print("buildDict("+str(dict_size)+", "+feature_type+", "+clustering_type+")")
    
    
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
        
        if len(des) > 20:
            np.random.shuffle(des)
            des = des[0:20] #arbitary keypoint number I chose 500
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

#DONE
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


#DONE
def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    
    classResult = np.array([])
    
    for image_scale in [8,16,32]:
        for K in [1,3,6]:
            start = timeit.default_timer()
            
            X_train = np.array([imresize(image, image_scale).flatten() for image in train_features])
            y_train = train_labels
            X_test = np.array([imresize(image, image_scale).flatten() for image in test_features])
            
            predicted_labels = KNN_classifier(X_train, y_train, X_test, K)
            accuracy = reportAccuracy(test_labels, predicted_labels)
            
            end = timeit.default_timer()
            
            classResult = np.append(classResult,[accuracy,end-start])

    return classResult
    

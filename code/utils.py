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
        return vocabulary

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary
    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    return Bow

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
    

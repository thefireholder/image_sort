import cv2
import numpy as np
import timeit
from sklearn import neighbors, svm, cluster

def add(a):
    x = a+4
    y = x/10
    return y

z = np.asarray([1,2,3])
y = add(z)
print(y)

def imresize(input_image, target_size):
    # resizes the input image to a new image of size [target_size, target_size]. normalizes the output image
    r_image = cv2.resize(input_image,(target_size,target_size)) #resized
    # to be zero-mean, and in the [-1, 1] range.
    output_image = cv2.normalize(r_image, None, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return output_image

image1 = cv2.imread('image1.png',cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('image2.png',cv2.IMREAD_GRAYSCALE)
z = np.array([image1, image2])
for n in z:
    y = imresize(n,30)
    print(y.shape)

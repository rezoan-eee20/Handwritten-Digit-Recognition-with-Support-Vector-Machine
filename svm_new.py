import cv2
# Import the modules
import matplotlib.pyplot as pt
import pandas as pd
from sklearn import metrics
from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn import preprocessing
import numpy as np
from collections import Counter
# Load the dataset
dataset=pd.read_csv("train.csv").as_matrix()
#dataset = datasets.fetch_mldata("MNIST Original")
# Extract the features and labels
features=dataset[0:21000,1:]
labels=dataset[0:21000,0]
#features = np.array(dataset.data, 'int16')
#labels = np.array(dataset.target, 'int')
# Extract the hog features
list_hog_fd = []
for feature in features:
  fd = hog(feature.reshape((28, 28)), orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
  list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')
# Normalize the features
pp = preprocessing.StandardScaler().fit(hog_features)
hog_features = pp.transform(hog_features)
print ("Count of digits in dataset", Counter(labels))
# Create an linear SVM object
clf = LinearSVC()
# Perform the training
clf.fit(hog_features, labels)
# Save the classifier
joblib.dump((clf, pp), "digits_cls.pkl", compress=3)

#Testing
# Read the input image
im = cv2.imread('test_image.png',cv2.IMREAD_COLOR)
# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)
# Threshold the image
ret, im_th = cv2.threshold(im_gray, 90, 255,cv2.THRESH_BINARY_INV)
# Find contours in the image
th,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# Get rectangles contains each contour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]
# For each rectangular region, calculate HOG features and predict
# the digit using classifier.
print(rects)

for rect in rects: 
   # Draw the rectangles
   cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
   # Make the rectangular region around the digit
   leng = int(rect[3] * 1.6)
   pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
   pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
   roi= im_th[pt1:pt1+leng, pt2:pt2+leng]
   if(roi.size==0):
     roi=np.zeros((50,50))
     roi = cv2.resize(roi,(28, 28),interpolation=cv2.INTER_AREA)
     roi = cv2.dilate(roi,(3, 3))
     # Calculate the HOG features
     roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
     roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
     nbr = clf.predict(roi_hog_fd)
     cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
   #print(roi)     
   # Resize the image
   else:
     roi = cv2.resize(roi,(28, 28),interpolation=cv2.INTER_AREA)
     roi = cv2.dilate(roi,(3, 3))
     # Calculate the HOG features
     roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
     roi_hog_fd = pp.transform(np.array([roi_hog_fd], 'float64'))
     nbr = clf.predict(roi_hog_fd)      
     cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
#check part
cv2.namedWindow("Resulting Image with Rectangular ROIs", cv2.WINDOW_NORMAL)
cv2.imshow("Resulting Image with Rectangular ROIs", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
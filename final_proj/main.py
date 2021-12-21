import numpy as np
import glob
import cv2
import numpy as np
import pickle
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from DataAugmentation import PreProcessor
from classifier import Classifier

obj_names = ['book','box', 'mug']
input_folder = 'imgs'
output_folder = 'dataset_augmented1'

# pre precess the image: 
#   it creates new images (256,256,3)
#   it creates the correspective lables: Lable1

# pre_processor = PreProcessor(obj_names, input_folder, output_folder)

""" Now we need to create our classifier """

svc = Classifier(output_folder, input_folder)
# from joblib import load

# svc = load('classification/modelSVM.joblib')

img = cv2.imread('try1.png')
print(svc.Predict(img, 11))
img = cv2.imread('try2.png')
print(svc.Predict(img, 11))
img = cv2.imread('try3.png')
print(svc.Predict(img, 11))
img = cv2.imread('try4.jpeg')
print(svc.Predict(img, 11))
img = cv2.imread('try5.png')
print(svc.Predict(img, 11))
img = cv2.imread('try6.jpg')
print(svc.Predict(img, 11))
img = cv2.imread('try7.jpg')
print(svc.Predict(img, 11))

# images = glob.glob(output_folder + '/*.jpg')
# assert images
# model_dataset = []
# for fname in sorted(images)
#     img = cv2.imread(fname)
#     h,w,d = img.shape
#     # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
#     model_dataset.append(img.reshape(h*w*d))

# np.savez("model_dataset", model_dataset)



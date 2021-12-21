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
import matplotlib.pyplot as plt

import os
os.environ['DISPLAY'] = ':0'

obj_name = ['book', 'mug'] #'box'

# images = glob.glob('dataset_augmented1/*.jpg')
# assert images
# ImageDataset_Augmented = []
# for fname in sorted(images):
#     img = cv2.imread(fname)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # img = cv2.resize(img,(256,256))
#     # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     h,w,d = img.shape
#     # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
#     ImageDataset_Augmented.append(img.reshape(h*w*d))

# ImageDataset_Augmented = np.asarray(ImageDataset_Augmented)
# # print(len(ImagDataset_Augmented))

# # Save the dataset
# np.savez("ImageDataset_Augmented1", ImageDataset_Augmented)
# Load the dataset
data = np.load("ImageDataset_Augmented1.npz")
ImageDataset_Augmented = data['arr_0']
data = np.load("Lable1.npz")
lable = data['arr_0']
# print(ImageDataset_Augmented.shape)

# print(len(ImagDataset_Augmented))
# print(len(lable))

# images = glob.glob('test/*.jpg')
# assert images
# TestImage = []
# for fname in sorted(images):
#     img = cv2.imread(fname)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(img,(100,100))
#     # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#     h,w,d = img.shape
#     # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
#     TestImage.append(img.reshape(h*w*d))

# # # Save the dataset
# np.savez("TestImage", TestImage)
# # # Load the dataset
# data = np.load("TestImage.npz")
# TestImage = data['arr_0']

# lable_train = np.loadtxt('./lable.txt', dtype = 'str')
# # lable_test = np.loadtxt('./lable1.txt', dtype = 'str')

X_train, X_test, lable_train, lable_test = train_test_split(ImageDataset_Augmented, lable, test_size = 0.2)

# Scale
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# PCA
pca = PCA(n_components = 30, svd_solver = 'randomized')
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# print(pca.explained_variance_ratio_)
# print(len(pca.explained_variance_ratio_))
# print(sum(pca.explained_variance_ratio_))


# SVM classification method
# svc = sk.svm.SVC(kernel='linear', C=1)
# svc.fit(X_train_pca, lable_train)

# # Save model in the current working directory
# model_name ="SVMmodel.pkl"
# with open(model_name, 'wb') as file:
#     pickle.dump(svc, file)

# # # Load from file
# # with open(pkl_filename, 'rb') as file:
# #     pickle_model = pickle.load(file)

# print("\nSVM\n")
# print("training score\n:", svc.score(X_train_pca, lable_train))
# print("testing score\n:", svc.score(X_test_pca, lable_test))

# img = cv2.imread('try1.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(svc.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try2.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(svc.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try3.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(svc.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try4.jpeg')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(svc.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try5.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(svc.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# print("test tesult:\n", classification_report(lable_test, y_predict, target_names = obj_name))
# print("confusion matrix:\n", confusion_matrix(lable_test, y_predict, labels = obj_name))

# # Apply Logistic Regression to the Transformed Data
# from sklearn.linear_model import LogisticRegression
# logisticRegr = LogisticRegression(solver = 'lbfgs', C = 1.0, max_iter = 10000)
# logisticRegr.fit(X_train_pca, lable_train)

# print("\nLogisticRegression\n")


# print("training score\n:", logisticRegr.score(X_train_pca, lable_train))
# print("testing score\n:", logisticRegr.score(X_test_pca, lable_test))


# img = cv2.imread('try1.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(logisticRegr.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try2.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(logisticRegr.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try3.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(logisticRegr.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try4.jpeg')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(logisticRegr.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# img = cv2.imread('try5.png')
# img = cv2.resize(img, (256,256))
# img = img.reshape(256*256*3)
# img = img.reshape(1,-1)
# img_scaled = scaler.transform(img)
# img_scaled_pca = pca.transform(img_scaled)
# i = 0
# result = []
# while i < 11:
#     result.append(logisticRegr.predict(img_scaled_pca))
#     i += 1
# if result.count('book') > 5:
#     y_predict = 'book'
# else: y_predict = 'mug'
# print(y_predict)

# # APPLY GRIDSEARCHCV
from sklearn.model_selection import GridSearchCV

param_grid = {'C': [1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1] }
clf = GridSearchCV(sk.svm.SVC(kernel='rbf'), param_grid = param_grid)
clf = clf.fit(X_train_pca, lable_train)

# print("\nGRIDSEARCH\n")
print("Best estimator found by grid search:")
print(clf.best_estimator_)


print("training score\n:", clf.score(X_train_pca, lable_train))
print("testing score\n:", clf.score(X_test_pca, lable_test))

img = cv2.imread('try1.png')
img = cv2.resize(img, (256,256))
img = img.reshape(256*256*3)
img = img.reshape(1,-1)
img_scaled = scaler.transform(img)
img_scaled_pca = pca.transform(img_scaled)
i = 0
result = []
while i < 11:
    result.append(clf.predict(img_scaled_pca))
    i += 1
if result.count('book') > 5:
    y_predict = 'book'
else: y_predict = 'mug'
print(y_predict)

img = cv2.imread('try2.png')
img = cv2.resize(img, (256,256))
img = img.reshape(256*256*3)
img = img.reshape(1,-1)
img_scaled = scaler.transform(img)
img_scaled_pca = pca.transform(img_scaled)
i = 0
result = []
while i < 11:
    result.append(clf.predict(img_scaled_pca))
    i += 1
if result.count('book') > 5:
    y_predict = 'book'
else: y_predict = 'mug'
print(y_predict)

img = cv2.imread('try3.png')
img = cv2.resize(img, (256,256))
img = img.reshape(256*256*3)
img = img.reshape(1,-1)
img_scaled = scaler.transform(img)
img_scaled_pca = pca.transform(img_scaled)
i = 0
result = []
while i < 11:
    result.append(clf.predict(img_scaled_pca))
    i += 1
if result.count('book') > 5:
    y_predict = 'book'
else: y_predict = 'mug'
print(y_predict)

img = cv2.imread('try4.jpeg')
img = cv2.resize(img, (256,256))
img = img.reshape(256*256*3)
img = img.reshape(1,-1)
img_scaled = scaler.transform(img)
img_scaled_pca = pca.transform(img_scaled)
i = 0
result = []
while i < 11:
    result.append(clf.predict(img_scaled_pca))
    i += 1
if result.count('book') > 5:
    y_predict = 'book'
else: y_predict = 'mug'
print(y_predict)

img = cv2.imread('try5.png')
img = cv2.resize(img, (256,256))
img = img.reshape(256*256*3)
img = img.reshape(1,-1)
img_scaled = scaler.transform(img)
img_scaled_pca = pca.transform(img_scaled)
i = 0
result = []
while i < 11:
    result.append(clf.predict(img_scaled_pca))
    i += 1
if result.count('book') > 5:
    y_predict = 'book'
else: y_predict = 'mug'
print(y_predict)

y_predict = clf.predict(X_test_pca)

# print("test tesult:\n", classification_report(lable_test, y_predict, target_names = obj_name))
# print("confusion matrix:\n", confusion_matrix(lable_test, y_predict, labels = obj_name))

# create a heatmap
# cm = confusion_matrix(lable_test, y_predict, labels = obj_name)
# plt.switch_backend('Qt4Agg') # to display the graph
# fig, ax = plt.subplots(figsize=(8, 8))
# ax.imshow(cm)
# ax.grid(False)
# ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted book', 'Predicted mug'))
# ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual book', 'Actual mug'))
# ax.set_ylim(1.5, -0.5)
# for i in range(2):
#     for j in range(2):
#         ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
# plt.show()

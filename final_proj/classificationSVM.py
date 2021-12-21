import numpy as np
import glob
import cv2
import numpy as np
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


def ReadTrainingImages(path):
    images = glob.glob(path + '/*.jpg')
    assert images
    
    # Read Images
    # ImageDataset = []
    # for fname in sorted(images):
    #     img = cv2.imread(fname)
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #     img = cv2.resize(img,(256,256))
    #     # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #     h,w,d = img.shape
    #     # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
    #     ImageDataset.append(img.reshape(h*w*d))

    # ImageDataset = np.asarray(ImageDataset)
    # print(len(ImageDataset))
    
    # # # # Save the dataset
    # np.savez("ImageDataset", ImageDataset)
    # # # # Load the dataset
    data = np.load("ImageDataset.npz")
    ImageDataset = data['arr_0']

    return ImageDataset


def LoadLableFile():
    # Import label
    lable = np.loadtxt('./lable.txt', dtype = 'str')
    return lable


def applyPCA(dataset, lable, n_components):
    # Split data
    X_train, X_test, lable_train, lable_test = train_test_split(dataset, lable, test_size = 0.2)

    # Scale
    scaler = StandardScaler()
    
    # X_train_scaled = StandardScaler().fit_transform(X_train)
    # X_test_scaled = StandardScaler().fit_transform(X_test)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # PCA
    pca = PCA(0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    print(pca.explained_variance_ratio_)
    print(len(pca.explained_variance_ratio_))
    print(sum(pca.explained_variance_ratio_))

    return X_train_pca, X_test_pca, lable_train, lable_test 


def ModifyNewImage(img):
    img = cv2.resize(img, (256,256))
    img = img.reshape(256*256*3)
    img = img.reshape(1,-1)
    img_scaled = scaler.transform(img)
    img_scaled_pca = pca.transform(img_scaled)
    return img_scaled_pca
    # y_predict = logisticRegr.predict(img_scaled_pca)
  

def main():

    path = './dataset'
    ImageDataset = ReadTrainingImages(path)
    lable = LoadLableFile()

    # Apply PCA
    X_train_pca, X_test_pca, lable_train, lable_test = applyPCA(ImageDataset, lable, 10)

    # SVM classification method
    svc = sk.svm.SVC(kernel='linear', C=1)
    svc.fit(X_train_pca, lable_train)

    y_predict = svc.predict(X_test_pca)

    print("training score\n:", svc.score(X_train_pca, lable_train))
    print("testing score\n:", svc.score(X_test_pca, lable_test))

    print("test tesult:\n", classification_report(lable_test, y_predict, target_names = obj_name))
    print("confusion matrix:\n", confusion_matrix(lable_test, y_predict, labels = obj_name))

    # create a heatmap
    cm = confusion_matrix(lable_test, y_predict, labels = obj_name)
    plt.switch_backend('Qt4Agg') # to display the graph
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cm)
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted book', 'Predicted mug'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual book', 'Actual mug'))
    ax.set_ylim(1.5, -0.5)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
    plt.show()


if __name__ == '__main__':
    main()




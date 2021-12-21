import cv2
import numpy as np 
import os
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def main ():

    train_path = './dataset'

    # read images for training
    images = getImages(train_path)
    # Create sift object
    sift = cv2.xfeatures2d.SIFT_create()
    descriptor_list = []
    train_labels = np.array([])
    label_count = 3
    image_count = len(images)
    
    # read all the images and classify them (add a label)
    for img_path in images:
        if("book" in img_path):
            class_index = 0
        elif("box" in img_path):
            class_index = 1
        elif("mug" in img_path):
            class_index = 2

        train_labels = np.append(train_labels, class_index)
        img = resizeImg(img_path)
        # get descriptor and keypoint of the img
        kp, des = sift.detectAndCompute(img, None)
        descriptor_list.append(des)

    # Stack descriptors vertically in a numpy array 
    descriptors = vstackDescriptors(descriptor_list)

    # Apply kmeans to the descriptors
    num_clusters = 200 # num of centroids
    n_init = 15 # number of times kmeans will run with different centroids
    kmeans = KMeans(n_clusters = num_clusters, n_init = n_init).fit(descriptors)

    # Extract features
    imgs_features = extractFeatures(kmeans, descriptor_list, image_count, num_clusters)

    # Scale features
    scaler = StandardScaler().fit(imgs_features)        
    imgs_features = scaler.transform(imgs_features)

    # Apply SVM to the features
    svm = applySVM(imgs_features, train_labels)

def getImages(train_path):
    images = []
    count = 0
    for folder in os.listdir(path):
        for file in  os.listdir(os.path.join(path, folder)):
            images.append(os.path.join(path, os.path.join(folder, file)))

    np.random.shuffle(images)    
    return images

def resizeImg(img_path):
    img = cv2.imread(img_path, 0)
    return cv2.resize(img,(256,256))

def vstackDescriptors(descriptor_list):
    descriptors = np.array(descriptor_list[0])
    for descriptor in descriptor_list[1:]:
        descriptors = np.vstack((descriptors, descriptor)) 
    return descriptors

def extractFeatures(kmeans, descriptor_list, image_count, num_clusters):
    img_features = np.array([np.zeros(num_clusters) for i in range(image_count)])
    for i in range(image_count):
        for j in range(len(descriptor_list[i])):
            feature = descriptor_list[i][j]
            feature = feature.reshape(1, 128)
            idx = kmeans.predict(feature)
            img_features[i][idx] += 1
    return img_features

def applySVM(imgs_features, train_labels):
    # define dict of parameters
    Cs = [0.5, 0.1, 0.15, 0.2, 0.3, 1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5]
    gammas = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 0.11, 0.095, 0.105]
    param_grid = {'kernel':('linear','rbf', 'sigmoid','precomputed','poly'), 'C': Cs, 'gamma' : gammas}
    # find the best ones
    grid_search = GridSearchCV(SVC(), param_grid, cv=nfolds)
    grid_search.fit(imgs_features, train_labels)
    print(grid_search.best_params_)
    # pick the best ones
    C, gamma, kernel = grid_search.best_params_.get("C"), grid_search.best_params_.get("gamma"), grid_search.best_params_.get("kernel")  

    svm = SVC(kernel = 'kernel', C =  C_param, gamma = gamma_param, class_weight = 'balanced')
    svm.fit(imgs_features, train_labels)
    print("training score\n:", clf.score(imgs_features, train_labels))
    # Save the trained model
    joblib.dump((clf, training_names, stdSlr, k), "BOVW.pkl", compress=3)    

    return svm


if __name__ == '__init__':
    main()
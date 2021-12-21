import numpy as np
import glob
import cv2
import numpy as np
import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
# from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


class Classifier(object):
    def __init__(self, img_path, lable_path):
        self.img_dataset = self.ReadImages(img_path)
        self.lable = self.ReadLables()
        self.n_components = 20
        self.obj_names = ['book', 'box', 'mug'] # 'box'
        self.applyPCA(self.img_dataset, self.lable, self.n_components)
        self.classify()
        # self.scaler = StandardScaler()
        # self.pca = PCA(n_components = self.n_components)


    def ReadImages(self, img_path):
        # images = glob.glob(img_path + '/*.jpg')
        # assert images
        
        # # Read Images
        # ImageDataset = []
        # for fname in sorted(images):
        #     img = cv2.imread(fname)
        #     h,w,d = img.shape
        #     # we need to flatten the image. we want a matrix like (n_imgaes, n_features)
        #     ImageDataset.append(img.reshape(h*w*d))

        # ImageDataset = np.asarray(ImageDataset)
        
        # # # # Save the dataset
        # np.savez("classification/flatten_dataset1", ImageDataset)
        # # # Load the dataset
        data = np.load("classification/flatten_dataset1.npz")
        ImageDataset = data['arr_0']
        return ImageDataset


    def ReadLables(self):# lable_path):
        data = np.load("classification/Lable.npz")
        return data['arr_0']


    def applyPCA(self, dataset, lable, n_components):   
        # Split data
        X_train, X_test, self.lable_train, self.lable_test = train_test_split(dataset, lable, test_size = 0.2)

        # Scale
        self.scaler = StandardScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # PCA
        self.pca = PCA(0.95)
        self.X_train_pca = self.pca.fit_transform(X_train_scaled)
        self.X_test_pca = self.pca.transform(X_test_scaled)

        # print(self.pca.explained_variance_ratio_)
        # print(len(self.pca.explained_variance_ratio_))
        # print(sum(self.pca.explained_variance_ratio_))

        # return X_train_pca, X_test_pca, lable_train, lable_test 


    def classify(self):

        # SVM classification method
        # self.svc = sk.svm.SVC(kernel='poly', C=1)
        # self.svc.fit(self.X_train_pca, self.lable_train)

        # APPLY GRIDSEARCHCV
        from sklearn.model_selection import GridSearchCV

        param_grid = {'C': [1e0, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5],
                    'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
        self.svc = GridSearchCV(sk.svm.SVC(kernel='rbf'), param_grid)
        self.svc = self.svc.fit(self.X_train_pca, self.lable_train)
        print("Best estimator found by grid search:")
        print(self.svc.best_estimator_)

        ## Apply Logistic Regression to the Transformed Data
        # from sklearn.linear_model import LogisticRegression
        # self.svc  = LogisticRegression(solver = 'lbfgs', C = 2.0, max_iter = 10000)
        # self.svc.fit(self.X_train_pca, self.lable_train)

        #save the model
        from joblib import dump, load
        dump(self.svc, 'classification/modelSVM.joblib')
        # clf = load('filename.joblib')
        # joblib.dump(clf, 'classification/ClassifierSVM.pkl') 
    
        # Show Result
        y_predict = self.svc.predict(self.X_test_pca)

        print("training score\n:", self.svc.score(self.X_train_pca, self.lable_train))
        print("testing score\n:", self.svc.score(self.X_test_pca, self.lable_test))

        # print("test tesult:\n", classification_report(self.lable_test, y_predict, target_names = self.obj_names))
        # print("confusion matrix:\n", confusion_matrix(self.lable_test, y_predict, labels = self.obj_names))



    def Predict(self, img, n_prediction):
        # resize image
        img = cv2.resize(img, (256,256))
        # flatten
        img = img.reshape(256*256*3)
        # convert to (1, n_pixel)
        img = img.reshape(1,-1)
        # scale image
        img = self.scaler.transform(img)
        # apply PCA
        img = self.pca.transform(img)
        # return the average prediction 
        idx = 0
        prediction = [0, 0, 0]
        lables = ['mug', 'box', 'book']
        while idx < n_prediction:
            predict = self.svc.predict(img)
            if predict == 'mug': prediction[0] += 1
            elif predict == 'box': prediction[1] += 1
            else: prediction[2] += 1
            idx += 1
        print(prediction)
        return lables[prediction.index(max(prediction))]


    
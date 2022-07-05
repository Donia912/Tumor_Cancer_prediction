# import libraries:
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np
# import files
from preprocessing import scaling
import joblib

class pca_algo:
     model_pca_filename = "pca_model.job_lib"
     def __init__(self, X_train, X_test):
         self.X_train = X_train
         self.X_test = X_test
     def pca_func(self):
        sc = scaling(self.X_train, self.X_test)
        sc.scale()
        pca = PCA(n_components=2)
        self.X_train = pca.fit_transform(sc.X_train)
        self.X_test = pca.transform(sc.X_test)
        explained_variance = pca.explained_variance_ratio_
        pca_quality = pca.singular_values_  #توضح مدى الكفاءة
        # print("X_train pca data:",self.X_train)
        print("explained_variance: \n", explained_variance)
        # print(pca_quality)
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X_train[:, 0], self.X_train[:, 1])
        plt.xlabel('First principle component')
        plt.ylabel('Second principle component')
        print(np.shape(self.X_train))
        plt.show()
        # save model
        joblib.dump(pca, self.model_pca_filename)

        # load model

     def pca_load (self):
         load_obj =joblib.load(self.model_pca_filename)
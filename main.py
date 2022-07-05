# Importing the required packages
import pandas as pd # read data
import numpy as np
from preprocessing import process
from Decision_tree_algorthim import Decision_Tree_algorithm
from regerssion import regression
from SVM_algorithm import svm_algorithm
from PCA import pca_algo


#  load_Data
dataset = pd.read_csv("Tumor Cancer Prediction_Data.csv")
# cleaning and split
pre_obj1 = process(dataset)
pre_obj1.clean()
pre_obj1.split_fun()

# PCA
pca_obj1=pca_algo(pre_obj1.x_train,pre_obj1.x_test)
pca_obj1.pca_func()


# decision tree
decision_obj1=Decision_Tree_algorithm(pca_obj1.X_train,pca_obj1.X_test,pre_obj1.y_train,pre_obj1.y_test)
decision_obj1.Decision()

# logistic regression
logistic_obj1=regression(pca_obj1.X_train,pca_obj1.X_test,pre_obj1.y_train,pre_obj1.y_test)
logistic_obj1.reg()
# svm_algo
svm_obj1 = svm_algorithm(pca_obj1.X_train,pca_obj1.X_test,pre_obj1.y_train,pre_obj1.y_test)
svm_obj1.classification()

# voting
def voting_fun():
    big_list = []
    for x in range(len(svm_obj1.y_pred_svm_test)):
        big_list.append(
            logistic_obj1.y_pred_logistic_test[x] + svm_obj1.y_pred_svm_test[x] + decision_obj1.y_pred_decision_test[x])

        if big_list[x] > 1:
            big_list.pop()
            big_list.append("Cancer")
        else:
            big_list.pop()
            big_list.append("Not Cancer")

    print(big_list)


voting_fun()

print("----------------------------------------------------------------------------------")
print("----------------------------------------------------------------------------------")
# input for new data
path = input("Enter your path of file :  ")
path1 = pd.read_csv(path)

# cleaning and split
pre_obj2 = process(path1)
pre_obj2.clean()

pre = pre_obj2.split_x_y()
pca_obj2=pca_algo(pre,pre)
pca_obj2.pca_func()

decision_obj1.load_decision(pca_obj2.X_test)
svm_obj1.load_svm(pca_obj2.X_test)
logistic_obj1.load_logistic(pca_obj2.X_test)


# PCA
# pca_obj2=pca_algo(pre_obj2.x_train,pre_obj2.x_test)
# pca_obj2.pca_func()


# # decision tree
# decision_obj2=Decision_Tree_algorithm(pca_obj2.X_train,pca_obj2.X_test,pre_obj2.y_train,pre_obj2.y_test)
# decision_obj1.Decision()

# # logistic regression
# logistic_obj2=regression(pca_obj2.X_train,pca_obj2.X_test,pre_obj2.y_train,pre_obj2.y_test)
# logistic_obj2.reg()
# # svm_algo
# svm_obj2 = svm_algorithm(pca_obj2.X_train,pca_obj2.X_test,pre_obj2.y_train,pre_obj2.y_test)
# svm_obj2.classification()

# # PCA
# # pca_obj2=pca_algo(pre_obj2.x_train,pre_obj2.x_test)
# # pca_obj2.pca_load()
#
#
# # decision tree
#
#
# # logistic regression
#
# # svm_algo
#
# decision_obj2 = Decision_Tree_algorithm(pre_obj1.x_train, pre_obj1.x_test, pre_obj1.y_train, pre_obj1.y_test)
# decision_obj2.load_decision(path1)
# svm_obj2 = svm_algorithm(pre_obj1.x_train, pre_obj1.x_test,pre_obj1. y_train, pre_obj1.y_test)
# svm_obj2.load_svm(path1)
# #
# #
# logistic_obj2 = regression(pre_obj2.x_train, pre_obj2.x_test, pre_obj2.y_train, pre_obj2.y_test)
# logistic_obj2.load_svm(path1)
# #

def voting_fun2():
    big_list = []
    for x in range(len(svm_obj1.result)):
        big_list.append(svm_obj1.result[x] + logistic_obj1.result[x] + decision_obj1.result[x])

        if big_list[x] > 1:
            big_list.pop()
            big_list.append("Cancer")
        else:
            big_list.pop()
            big_list.append("Not Cancer")

    print(big_list)


voting_fun2()
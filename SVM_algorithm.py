#(1)import libraries:
from sklearn.svm import SVC
from sklearn import metrics
import joblib


#creating_model svm
class svm_algorithm:
    model_SVM_filename = "svm_model.job_lib"
    result = []
    y_pred_svm_test = []
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def classification(self):
        # classify the model
        clf = SVC(kernel='linear', C=1)
        # train the model
        clf.fit(self.x_train, self.y_train)


        # predict on the x_test
        self.y_pred_svm_test = clf.predict(self.x_test)
        svm_accuracy =metrics.accuracy_score(self.y_test,  self.y_pred_svm_test)

        # print("the precission of the svm algorithm : ",metrics.precision_score(self.y_test,y_pred))
        # print("the Recall of the svm algorithm : ",metrics.recall_score(self.y_test,y_pred))

        # save modle
        joblib.dump(clf, self.model_SVM_filename)
        print('Model svm is saved into to disk successfully')
        print(end='\n')

        print("SVM MODEL :: ")
        print(end='\n')
        print("predict is : ", self.y_pred_svm_test)
        print(end='\n')

        print("accuracy is : ", svm_accuracy)
        print(end='\n')

        print("----------------------------------------------------------------")

        # load model

    def load_svm(self,test1):
        svm_model = joblib.load(self.model_SVM_filename)
        self.result = svm_model.predict(test1)
        print("svm predict result is :: ")
        print(end='\n')
        print("Svm predict is : ", self.result)
        print(end='\n')
        print("Svm Accuracy : ", svm_model.score(self.x_test, self.y_test))
        print(end='\n')
        print(".................................................................")
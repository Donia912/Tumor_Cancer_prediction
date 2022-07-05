# Importing the required package
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import joblib

class Decision_Tree_algorithm:
    model_decision_filename = "decision_model.job_lib"
    result=[]
    y_pred_decision_test = []
    def __init__(self, x_train, x_test, y_train, y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def Decision(self):
        # take an object
        classifier = DecisionTreeClassifier()
        # train the model using training set
        classifier.fit(self.x_train, self.y_train)
        # Predict the response for test dataset
        self.y_pred_decision_test= classifier.predict(self.x_test)
        # Model Accuracy: how often is the classifier correct?

        # Model Precision: what percentage of positive tuples are labeled as such
        decisiontree_accuracy = metrics.accuracy_score(self.y_test,self.y_pred_decision_test)

        # save model

        joblib.dump(classifier, self.model_decision_filename)
        print(end='\n')
        print('Model decision is saved into to disk successfully')
        print(end='\n')

        print("DECISION TREE MODEL ::  ")
        print(end='\n')
        print("predict is : ", self.y_pred_decision_test)
        print(end='\n')

        print("accuracy is  : ", decisiontree_accuracy)
        print(end='\n')

        print("-------------------------------------------------------------")

        # load model
    def load_decision(self,test1):
        decision_model = joblib.load(self.model_decision_filename)
        self.result = decision_model.predict(test1)
        print("Decision predict is :: ")
        print(end='\n')
        print("Decision predict is : ", self.result)

        print(end='\n')
        print("Decision tree Accuracy : ", decision_model.score(self.x_test, self.y_test))
        print(end='\n')
        print("...................................................................")
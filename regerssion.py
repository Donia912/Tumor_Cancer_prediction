from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import joblib
class regression:
    model3_filename = "logistic_model_job_lib.job_lib"
    result=[]
    y_pred_logistic_test = []


    def __init__(self,x_train,x_test,y_train,y_test):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test



    def reg(self):
        logistic_model = LogisticRegression(max_iter=2000)
        #Fit the model
        logistic_model.fit(self.x_train, self.y_train)
        #Predict the output
        self.y_pred_logistic_test = logistic_model.predict(self.x_test)
        # calculate the accuracy
        score = logistic_model.score(self.x_test,self.y_pred_logistic_test)
        logisticAccuracy=metrics.accuracy_score(self.y_test,self.y_pred_logistic_test)
        # save model
        joblib.dump(logistic_model, self.model3_filename)
        print(end='\n')
        print('Model logistic is saved into to disk successfully')

        print(end='\n')
        print("LOGISTIC MODEL :: ")
        print(end='\n')
        print("predict is : ",   self.y_pred_logistic_test )
        print(end='\n')
        print("score is : ",   score)
        print(end='\n') 
        print("Accuracy is : ",   logisticAccuracy)
        print(end='\n') 
        print("----------------------------------------------------------------")

# load
    def load_logistic(self,test1):
        logistic_model = joblib.load(self.model3_filename)
        self.result = logistic_model.predict(test1)
        print("Logistic predict result is ::  ")
        print(end='\n')
        print("Logistic predict is : ", self.result)
        print(end='\n')
        print("Logistic Accuracy : ", logistic_model.score(self.x_test, self.y_test))
        print(end='\n')
        print("...................................................................")
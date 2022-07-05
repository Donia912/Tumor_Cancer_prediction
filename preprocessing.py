# import libraries:

from sklearn.preprocessing import StandardScaler # for scaling
from sklearn.model_selection import train_test_split



class process:
    def __init__(self,dataset):
        self.dataset = dataset

    def clean(self):
        # # (3.1)format >>>
        # print("before convert:")
        # print(data["diagnosis"])                                         # the data type of the column before convert
        self.dataset["diagnosis"] = self.dataset["diagnosis"].replace({"B": 1, "M": 0})  # converting last column to 0&1
        # print("after convert:")
        # print(data["diagnosis"])                                          # the data type of the column after convert

        ##(3.2)null>>>

        # print("summution of null", self.data.isnull().sum())        # sum of null cels
        ###first way:(by droping rows which have null values but it is a bad way because sometimes it delete lots rows so i don't have enough data)
        # self.data.dropna(how='any', inplace=True, axis=0)        #how >>>> ليها قميتين all(لما يكون ال الصف كله ب نال او العمود او الجدول)/any(at least cill).....inplace >> (default>>false)  If you want to change the original DataFrame, use the inplace = True argument

        ###second way:(replace null valuse with the mean of the column and it is the best way in our case)
        columns = self.dataset.columns.values  # view columns's names in array of list

        for i in columns:
            if i == "Index":
                continue
            else:
                nullVal = self.dataset[i].mean()
                # print(i)
                self.dataset[i].fillna(nullVal, inplace=True)

        # # (3.3)duplicated >>>

        # # self.data.duplicated()                             # this function is used for discovering duplicates

    def split_x_y(self):
        self.dataset = self.dataset.dropna()
        missing = self.dataset.isnull().any(axis=1)
        dataset_input = self.dataset.drop(columns=['diagnosis', 'Index'])
        dataset_output = self.dataset['diagnosis']
        dataset_input.drop_duplicates(inplace=True)  # remove duplicated data (True>>> remove from the exicel sheet)
        return dataset_input
    def split_fun(self):
        # (4) split data to features and label:

        X = self.dataset.iloc[:, 1:31]  # Features>>from f1 to f30
        Y = self.dataset['diagnosis']  # Label>> it is What to expect
        self.dataset.drop_duplicates(inplace=True)  # remove duplicated data (True>>> remove from the exicel sheet)

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(X, Y, test_size=0.25,
                                                                                random_state=0)
        print(end='\n')
        print(self.x_train.shape)
        print(self.y_train.shape)
        print("-----------------")
        print(self.x_test.shape)
        print(self.y_test.shape)
        print(end='\n')


class scaling:
    def __init__(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test

    # Standard Scaler for Data:
    def scale(self):
        scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

        # showing data
        # print('X_train \n', self.X_train[:3])
        # print('X \n_test',  self.X_test[:3])
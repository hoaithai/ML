
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split


DATASET_TRAIN_PATH = "C:/Users/sev_user/Downloads/TempData/train1.csv"
DATASET_TEST_PATH = "C:/Users/sev_user/Downloads/TempData/test1.csv"

def main():
    data_features = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "Sibsp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

    data_train = pd.read_csv(DATASET_TRAIN_PATH, names=data_features)
    data_test = pd.read_csv(DATASET_TEST_PATH, names=data_features)

    #print "Headers: ", data_train.colums.values
    print(data_train[data_features[:-1]])
    #print(data_train[data_features[1]])
    train_x = data_train[data_features[:-1]]
    train_y = data_train[data_features[1]]

    test_x = data_test[data_features[:-1]]
    #test_y = data_test[data_features[0]]

   #test_x, test_y = train_test_split(data_test[data_features[:-1]])
    #print "Train data is:"
   # print train_x


    lr = linear_model.LinearRegression()
    lr.fit(train_x, train_y)

   # print "Logistic regression Test Accuracy :: ", metrics.accuracy_score(test_y, lr.predict(test_x))
    #test_y = lr.predict(test_x)
    #print "Outcome:: ", lr.predict(test_x)

'''
    df = pd.DataFrame(test_y)
    df.to_csv("D:/Data/results.csv")
    '''

if __name__ == "__main__":
    main()



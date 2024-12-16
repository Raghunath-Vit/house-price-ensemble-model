import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn as sk
import pickle

heart_data = pd.read_csv('D:\python\heart.csv')

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

#apply the knn method
Knn = KNeighborsClassifier(n_neighbors = 2)

#train the data
mod=Knn.fit(X_train,Y_train)
pickle.dump(mod, open('model.pkl', 'wb'))



#test the data
accuracy = Knn.score(X_test, Y_test)#this to see how accurate the algorithm is in terms
#of defining the diabetes to be either 1 or 0
print('accuracy of the model is: ', accuracy)

input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = Knn.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')
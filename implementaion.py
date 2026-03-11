import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from svm import SVM_classifier


# loading the data from csv file to pandas dataframe
diabetes_data = pd.read_csv('/content/diabetes.csv')


# print the first 5 rows of the dataframe
diabetes_data.head()


size = diabetes_data.shape


value_count = diabetes_data['Outcome'].value_counts()


# separating the features and target
features = diabetes_data.drop(columns='Outcome', axis=1)
target = diabetes_data['Outcome']


scaler = StandardScaler()
scaler.fit(features)


StandardScaler(copy=True, with_mean=True, with_std=True)


standardized_data = scaler.transform(features)
features = standardized_data
target = diabetes_data['Outcome']

X_train, X_test, Y_train, Y_test = train_test_split(features, target, test_size=0.2, random_state = 2)
print(features.shape, X_train.shape, X_test.shape)

classifier = SVM_classifier(learning_rate=0.001, no_of_iterations=1000, lambda_parameter=0.01)
classifier.fit(X_train, Y_train)


X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, X_train_prediction)


X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, X_test_prediction)



input_data = (5,166,72,19,175,25.8,0.587,51)

# change the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

# standardizing the input data
std_data = scaler.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if (prediction[0] == 0):
  print('The person is not diabetic')
else:
  print('The Person is diabetic')









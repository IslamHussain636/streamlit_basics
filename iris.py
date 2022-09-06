import streamlit as st 
x = 4
st.write(x, 'squared is', x * x)

x = st.slider('x')  
# 👈 this is a widget
#st.write(x, 'squared is', x * x)

from sklearn.datasets import load_iris
iris= load_iris()
# Store features matrix in X
X= iris.data
#Store target vector in 
y= iris.target
# Names of features/columns in iris dataset
print(iris.feature_names)

# Names of target/output in iris dataset
print(iris.target_names)

# size of feature matrix
print(iris.data.shape)

# size of target vector
print(iris.target.shape)

#Import the classifier
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()

knn.fit(X, y)

#Predicting output of new data
knn.predict([[3.2, 5.4, 4.1, 2.5]])

# instantiate the model 
knn = KNeighborsClassifier(n_neighbors=1)# fit the model with data
knn.fit(X, y)# predict the response for new observations
knn.predict([[3, 5, 4, 2]])

# instantiate the model 
knn = KNeighborsClassifier(n_neighbors=5)# fit the model with data
knn.fit(X, y)# predict the response for new observations
knn.predict([[3, 5, 4, 2]])









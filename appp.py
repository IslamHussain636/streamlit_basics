import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

#heading 
st.write('''*** 
         # Exploaring different ml models and datasets
         Lets see which one is the best?
         ***''')

#creating a sidebar for dataset selection
dataset_name= st.sidebar.selectbox(
    'Select Datasets',
    ('iris','wine','breast_cancer')
)
# ml model selection 
classifier_name=st.sidebar.selectbox('Select Classifier',
                               ('SVM','KNN','Random Forest'))

# loading the dataset
def get_datasets(dataset_name):
    data=None
    if dataset_name=='iris':
        data=datasets.load_iris()
    elif dataset_name=='wine':
        data= datasets.load_wine()
    else :
         data=datasets.load_breast_cancer()
    x= data.data
    y=data.target 
    return x,y

# equalize the varible 
x,y=get_datasets(dataset_name)

#printing the shape of the datasets 
st.write('Shape of the dataset:',x.shape)
st.write('number of classes:',len(np.unique(y)))

#defining the parameters for the model:

def add_parameter_ui(classifier_name):
    params=dict() # define a dictionary
    if classifier_name=='SVM':
        C=st.sidebar.slider('C',0.01,10.0)
        params['C']=C
    elif classifier_name=='KNN':
        K=st.sidebar.slider('K',1,50)
        params['K']=K
    else:
        max_depth=st.sidebar.slider('max_depth',2,20)
        params['max_depth']=max_depth
        n_estimators=st.sidebar.slider('n_estimators',1,100)
        params['n_estimators']=n_estimators        
 # calling our function   
params=add_parameter_ui(classifier_name)


# parameters
def get_classifier(classifier_name,params):
    if classifier_name=='SVM':
        classifier=SVC(C=params['C'])
    elif classifier_name=='KNN':
        classifier=KNeighborsClassifier(n_neighbours=params['K'])
    else:
        classifier=RandomForestClassifier(n_estimators=params['n_estimators'],max_depth=params['max_depth'])
    return classifier

#calling our function
classifier=get_classifier(classifier_name,params)

if st.checkbox('show code'):
    with st.echo():
        
        #spliting the data into train and test
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1234)
        classifier.fit(x_train,y_train)
        y_pred=classifier.predict(x_test)

        # Accuracy 
        acc=accuracy_score(y_test,y_pred)
        st.write(f'Classifier= {classifier_name}')
        st.write(f'Accuracy ='acc)


#plot 
pca=PCA(2)
x1=x_projrcted[:,0]
x2=X_projected[:,1]
fig=plt.figure()
plt.scatter(x1,x2,c=y,alpha=0.8,cmap='viridis')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

plt.colorbar()
# show
st.pyplot(fig)



# pipreqs use to genereate requirments.txt file




  
 

from email import header
from statistics import mode
import streamlit as st
import seaborn as sns
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error , mean_squared_error,r2_score

# make containers 
header=st.container()
datasets= st.container()
features= st.container()
model_training= st.container()

with header:
    st.title('Kashti App')
    st.text('In this project we will work on titanic dataset')
    
with datasets:
    st.header('kashti doob gaie ooo')
    st.text('kashti wala data set')
    # import data set
    df=sns.load_dataset('titanic')
    df=df.dropna()
    st.write(df.head())
    
    st.subheader('oh bhai kitny admi ty')
    st.bar_chart(df['sex'].value_counts())
    # bar chart
    st.subheader('class wise bandy')
    st.bar_chart(df['class'].value_counts())
    
    st.bar_chart(df['age'].sample(10)) # or head(10)
    
    
with features:
    st.header('Hamry app ki feature')
    st.text('awi waly feature')
    st.markdown('1.**Feature 1** This will be a simple feature')
    st.markdown('2.**Feature 1** This will be a simple feature')
    st.markdown('3.**Feature 1** This will be a simple feature')

with model_training:
    st.header('kashti walo ka kia bana ') 
    st.text('Kitny bachgiey ty')           
    # making columns 
    input, display= st.columns(2)
    # pehly column mn selection points hoo
    max_depth=input.slider('how many people do you know ?',min_value=10, max_value=100, value=20, step=5)
  # n estimators 
n_estimators=input.selectbox('how many tress',options=[50,100,200,300,'NO limit'])  

# adding list of features 
input.write(df.columns)
# input feaure from user 
input_feature= input.text_input('which input we should use ?')


# machine learning model
model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
# apply a condition 
if n_estimators=='NO limit':
        model=RandomForestRegressor(max_depth=max_depth)
        
else:
        model= RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)    
# define input and output
X=df[[input_feature]]
y=df[['fare']]

#fit model
model.fit(X,y)

#prdiction
pred=model.predict(y)

# dispaly metrice
display.subheader('mean absolute error of the modes is ')
display.write(mean_absolute_error(y,pred))
display.subheader('mean squred error of the modes is ')
display.write(mean_squared_error(y,pred))
display.subheader('r2 score  of the modes is ')
display.write(r2_score(y,pred))


 


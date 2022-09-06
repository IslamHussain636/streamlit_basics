import streamlit as st
st.header('Streamlit is awesome with babay Aammar') 
st.text('bara mazay ki cheez hy ye streamlit Yra')

st.header('Idr kuch bi chaliengaa')

import seaborn as sns
df= sns.load_dataset('Iris')

st.write(df[['species','petal_length','sepal_length']].head())
st.bar_chart(df[['petal_length']])
st.line_chart(df[['petal_length']])
st.line_chart(df[['sepal_length']])
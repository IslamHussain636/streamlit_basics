import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
# title 
st.markdown('''
  # ** Exploratory Data Analysis web application **
  This is developed by Islam Hussain with codanics **EDA app**
  ''')
#how to upload your data file 
with st.sidebar.header("Upload your dataset(.csv)"):
    upload_file=st.header.file_uploader("upload your dataset", type=["csv"])
    df=sns.load_dataset('titanic')
    st.sidebar.markdown('[Example CSV file](df)')
    #st.sidebar.markdown('[Example CSV file](df)') here we can pass link of any dataset
# profilig report for pandas
if upload_file is not None:
  def load_csv():
    csv=pd.read_csv(upload_file)
    return csv 
  df=load_csv()
  pr=ProfileReport(df,explorative=True)
  st.header('**Input DF**')
  st.write(df)
  st.write('---')
  st.header('**Profile Report with pandas **')
  st_profile_report(pr) 
else:
  st.info("Awating for upload your dataset?")
  if st.but('PRESS to use example data'):
      # example dataset
      @st.cache()
      def load_data():
        a=pd.DataFrame(np.random.rand(100,5),
                       columns=['age','banana','codanics','Islam','EDA'])
        return a
      df=load_data()
      pr=ProfileReport(df,explorative=True)
      st.header('**Input DF**')
      st.write(df)
      st.write('---')
      st.header('**Profile Report with pandas **')
      st_profile_report(pr)
      
                           
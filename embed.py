import streamlit as st 
from streamlit_embedcode import github_gist
link="https://gist.github.com/islamhussain/f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8f8" # here we can pass link of any gist

st.write('Embed github gist:')
github_gist(link)
 
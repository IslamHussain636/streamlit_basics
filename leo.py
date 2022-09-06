# importing libraries 
import streamlit as st
import pandas as pd
from PIL import Image

st.write('''**image and video in streamlit app
         **''')
image1=Image.open('image1.jpg')
st.image(image1,width=300)

#add video 
st.write('''**video in streamlit app
         **''')
video1=open('leo.mp4','rb')
st.video(video1)

#audio in streamlit app
st.write('''**audio in streamlit app**''')
audio1=open('leo.mp3','rb')
st.audio(audio1)

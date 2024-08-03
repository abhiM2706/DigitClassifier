import streamlit as st
import pandas as pd
import math
from pathlib import Path
from openai import AzureOpenAI
from ChatGptAPI import GPT
from PIL import Image
from TransformImage import predict

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Digit Classifier',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

def initialize():
    if 'key' not in st.session_state:
        st.session_state['key'] = 'NN'

if(st.button("Use ChatGPT", type="primary")):
    initialize()
    st.session_state['key'] = 'GPT'
    print("changed to gpt")
    st.write("Upload Image to be classified by chatGPT 4.0")

if(st.button("Use Trained Neural Network", type='primary')):
    initialize()
    st.session_state['key'] = 'NN'
    print("changed to nn")
    st.write("Upload Image to be classified by NN")

img_file_buffer = st.file_uploader('Upload a PNG image', type='png')
if img_file_buffer is not None:
    image = Image.open(img_file_buffer)

    if(st.session_state['key']=='GPT'):
        gpt = GPT()
        answer = gpt.predict(img_file_buffer.name)
        st.write(answer)
    else:
        answer = predict(img_file_buffer.name)
        print(answer[0].item())
        st.write("The image seems to be the number "+ str(answer[0].item()))

    

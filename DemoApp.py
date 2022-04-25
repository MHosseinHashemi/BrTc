from re import X
from keras.models import load_model
import streamlit as st
from io import BytesIO
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import numpy as np



# Load model
my_model = load_model('effnet.h5')

# UI Design
st.set_page_config(layout='wide')
st.sidebar.markdown('This Web-Application allows you to classify the tumor from an MRI Pic')
st.header('Tumur Classifier')


# a function to choose the right label to show
def labelizer(p):
    if p==0:
        return 'Glioma Tumor'
    elif p==1:
        return 'There is no tumor'
    elif p==2:
        return 'Meningioma Tumor'
    else:
        return 'Pituitary Tumor'



# a function to manipulate the input pic
def classifier(image, model):
    image = cv2.resize(image, (150,150))
    image = image.reshape(1,150,150,3)
    # predicting the label
    prediction = model.predict_on_batch(image)
    # map the predition to labels
    classification = np.where(prediction == np.max(prediction))[1][0]
    output = str(prediction[0][classification]*100) + "% Confidence This is " + labelizer(classification)

    return output

    



input_file = st.file_uploader("Upload MRI pic", type=['jpg','png'])
if input_file is None:
    st.text("Please upload a picture")
else:
    img = Image.open(input_file)
    st.image(img, use_column_width=True)
    pred = classifier(img, my_model)
    st.success(pred)
    







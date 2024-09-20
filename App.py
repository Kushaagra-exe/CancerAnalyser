import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle as pkl





# batch_size = 32 

def load_model(model):
    with open(model, 'rb') as f:
        model = pkl.load(f)
    return model


def preprocess_image(image, img_size):
    image = tf.image.resize(image, img_size)
    image = image / 255.0
    return image

def load_and_preprocess_image(uploaded_file, img_size):
    image_bytes = uploaded_file.read()
    image = tf.io.decode_image(image_bytes, channels=3)
    return preprocess_image(image, img_size)


st.sidebar.title("Select your Model")

option = st.sidebar.selectbox(
    'Choose an option:',
    ['Lung Cancer', 'Colon Cancer', 'Brain Tumor']
)
if option == 'Lung Cancer':
    path = 'lung.pkl'
    img_size = (224, 224)
    class_names = ['Cancer not detected', 'Lung_Adenocarcinoma', 'Lung_Squamous_Carcinoma']
    st.sidebar.write("Accuracy: 92.15%")
    st.sidebar.write("F1 Score: 0.924")
    st.sidebar.write("Recall: 0.921")



elif option == 'Colon Cancer':
    path = 'colon_model.pkl'
    img_size=(180, 180)
    class_names = ['Colon adenocarcinoma', 'Colon benign tissue']
    st.sidebar.write("Accuracy: 96.39%")
    st.sidebar.write("F1 Score: 0.964")
    st.sidebar.write("Recall: 0.964")

elif option == 'Brain Tumor':
    path = 'brain_tumour.pkl'
    img_size = (224, 224)
    class_names = ['Glioma', 'Meningioma', 'No_Tumor' , 'Pituitary']
    st.sidebar.write("Accuracy: 89.43%")
    st.sidebar.write("F1 Score: 0.906")
    st.sidebar.write("Recall: 0.905")


model = load_model(path)





st.title("Cancer Analyser")

uploaded_file = st.file_uploader("Choose an image file...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = load_and_preprocess_image(uploaded_file, img_size)
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)

    st.subheader('Predication')
    # st.write(predictions)
    predicted_index = np.argmax(predictions)
    st.write(class_names[predicted_index], " : ", predictions[0][predicted_index]*100, "%")
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)



import streamlit as st
import numpy as np
import time
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model('/home/ujjwal/Programs/ML-DS/PROJECTS/Dog vs Cat/cat_dog_classifier.keras')

st.title('Dog vs Cat Classifier')
img = st.file_uploader('Upload Image', type=('png','jpg','jpge'))
resized_image=None


if img is not None:
    c1, c2 = st.columns(2)
    with c1:
        image = Image.open(img)
        resized_image = image.resize((256, 256))
        st.image(resized_image,width=200)
    with c2:
        st.write("**Prediction value ranges from 0 to 1.**")
        st.text("Value closer to 0: Cat")
        st.text("Value closer to 1: Dog")
        st.write("[0-----------------------0.5------------------------1]")
        st.write("[Cat---------------------------------------------Dog]")

    if st.button('Classify') and image is not None:
        
        # Preprocess the image for prediction
        image_array = np.array(resized_image) / 255.0  # Normalizing pixel values to [0, 1]
        image_array = np.expand_dims(image_array, axis=0)  # Adding batch dimension

        # Predict using the model
        prediction = model.predict(image_array)

        with st.spinner("Please Wait..."):
            time.sleep(3)
        # Display the result
        if prediction[0][0] > 0.5:
            st.success("It's a **Dog! 🐶**")
        else:
            st.success("It's a **Cat! 🐱**")
        st.write("Prediction value:",prediction[0][0])
        

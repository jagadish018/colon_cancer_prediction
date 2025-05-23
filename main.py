import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown
import zipfile

# Google Drive zipped model setup
file_id = '1Izx86aritVv9VgHCf9rh6UxGDZv0wob4'
zip_path = 'model.zip'
model_folder = 'saved_model'

# Download and unzip model if not present
if not os.path.exists(model_folder):
    with st.spinner('Downloading model... Please wait...'):
        gdown.download(f'https://drive.google.com/uc?id={file_id}&export=download', zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(model_folder)
    os.remove(zip_path)

@st.cache_resource(show_spinner=False)
def load_model():
    model = tf.keras.models.load_model(model_folder)
    return model

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(256, 256))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Cancer Prediction"])

# Main Page
if app_mode == "Home":
    st.header("COLON CANCER PREDICTION SYSTEM")
    image_path = "images.jpeg"
    
    if os.path.exists(image_path):
        st.image(image_path, use_column_width=True)
    else:
        st.warning("Image not found. Please ensure 'images.jpeg' is in the correct directory.")
    
    st.markdown("""
    Welcome to the Colon Cancer Prediction System! 🩺🔍

    Our mission is to assist in the early detection of colon cancer using advanced deep learning techniques. Upload a histopathological image, and our system will analyze it to detect any signs of colon cancer. Together, let's improve diagnosis and treatment outcomes!

    ### How It Works
    1. *Upload Image:* Go to the *Cancer Prediction* page and upload a histopathological image.
    2. *Analysis:* Our system will process the image using advanced algorithms to identify potential cancerous regions.
    3. *Results:* View the results and recommendations for further action.

    ### Why Choose Us?
    - *Accuracy:* Our system utilizes state-of-the-art ResNet50 model for accurate cancer detection.
    - *User-Friendly:* Simple and intuitive interface for seamless user experience.
    - *Fast and Efficient:* Receive results in seconds, enabling timely medical intervention.

    ### Get Started
    Click on the *Cancer Prediction* page in the sidebar to upload an image and experience the power of our Colon Cancer Prediction System!

    ### About Us
    Learn more about the project, our team, and our goals on the *About* page.
    """)

# About Project
elif app_mode == "About":
    st.header("About Us")
    st.markdown("""
    ### Welcome to our project website, dedicated to advancing medical technology through innovative solutions!

    We are a team of dedicated individuals with a deep passion for data science and medical research. Driven by our keen interest in harnessing the power of technology to solve real-world health problems, we have embarked on a mission to improve the early detection of colon cancer.

    Our project focuses on leveraging the ResNet50 deep learning model to accurately identify colon cancer from histopathological images. By providing a reliable and efficient diagnostic tool, we aim to support healthcare professionals in their efforts to diagnose and treat colon cancer early, ultimately improving patient outcomes.

    This project has been a journey of discovery and innovation, guided by the invaluable mentorship of our advisors. We extend our heartfelt thanks to them for their unwavering support and guidance, which have been instrumental in the successful realization of our vision.

    Our website is designed with the hope of empowering medical professionals by providing them with advanced tools and insights for better patient management. Through our platform, we aspire to contribute to the medical community by offering practical solutions that lead to more efficient and effective healthcare practices.

    Thank you for visiting our website. We are excited to share our work with you and hope it makes a meaningful impact in the field of healthcare. Together, let's create a future where technology and medicine go hand in hand for a healthier tomorrow.
    """)

# Prediction Page
elif app_mode == "Cancer Prediction":
    st.header("Cancer Prediction")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, use_column_width=True)
        
        if st.button("Predict"):
            st.write("Our Prediction")
            result_index = model_prediction(test_image)
            
            class_name = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
            result_label = class_name[result_index]
            
            if result_label in ['DEB', 'LYM', 'MUC', 'MUS', 'STR', 'TUM']:
                st.error(f"The image is a cancer tissue. It's a {result_label}.")
            elif result_label == 'NORM':
                st.success("The image is not a cancer tissue.")
            else:
                st.warning(f"The image is not a cancer tissue. It is another object: {result_label}")

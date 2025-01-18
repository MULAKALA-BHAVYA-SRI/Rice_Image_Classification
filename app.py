import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image, ImageOps

# Load models
resnet_model = load_model('resnet_model.h5')
vgg16_model = load_model('vgg16_model.h5')
xception_model = load_model('xception_model.h5')
cnn_model = load_model('Model.h5')  # Load your CNN model

# Define image size for each model
resnet_img_size = (50, 50)
vgg16_img_size = (50, 50)
#xception_img_size = (71, 71)
cnn_img_size = (50, 50)  # CNN model uses 50x50 image size

# Mapping of model outputs (update based on your classes)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Function to preprocess and predict using a model
def predict_image(model, img, img_size):
    # Convert to RGB (in case the image has an alpha channel)
    img = img.convert("RGB")
    
    # Resize the image to match model input size
    img = img.resize(img_size)
    
    # Convert image to numpy array
    img_array = np.array(img)
    
    # Normalize image data to [0, 1] range
    img_array = img_array / 255.0
    
    # Add batch dimension (model expects a batch, even if it's just one image)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Make prediction
    prediction = model.predict(img_array)
    return prediction

# Streamlit UI
st.title("Rice Variety Classification")

st.markdown("""
This app allows you to classify rice varieties based on images. 
You can upload an image of rice, and the model will predict the variety.
""")

# File uploader widget
uploaded_file = st.file_uploader("Choose a rice image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Select model for prediction
    model_option = st.selectbox("Choose model", ["ResNet", "VGG16", "CNN"])
    
    if model_option == "ResNet":
        st.write("Using ResNet model for classification...")
        prediction = predict_image(resnet_model, img, resnet_img_size)
    elif model_option == "VGG16":
        st.write("Using VGG16 model for classification...")
        prediction = predict_image(vgg16_model, img, vgg16_img_size)
    #elif model_option == "Xception":
       # st.write("Using Xception model for classification...")
       # prediction = predict_image(xception_model, img, xception_img_size)
    elif model_option == "CNN":
        st.write("Using CNN model for classification...")
        prediction = predict_image(cnn_model, img, cnn_img_size)

    # Get the predicted class
    predicted_class = class_names[np.argmax(prediction)]
    st.write(f"Predicted rice variety: {predicted_class} with confidence: {np.max(prediction) * 100:.2f}%")

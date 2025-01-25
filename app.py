import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load models
resnet_model = load_model('resnet_model.h5')
vgg16_model = load_model('vgg16_model.h5')
cnn_model = load_model('Model.h5')  # Load your CNN model

# Define image size for each model
resnet_img_size = (50, 50)
vgg16_img_size = (50, 50)
cnn_img_size = (50, 50)  # CNN model uses 50x50 image size

# Mapping of model outputs (update based on your classes)
class_names = ['Arborio', 'Basmati', 'Ipsala', 'Jasmine', 'Karacadag']

# Descriptions for each rice variety
descriptions = {
    'Arborio': "Arborio rice is a short-grain variety primarily used in risottos due to its creamy texture. It is rich in starch, allowing it to absorb flavors effectively, making it ideal for rich and savory dishes in Italian cuisine.",
    'Basmati': "Basmati rice is a long-grain aromatic variety known for its distinct fragrance and flavor. Commonly used in Indian and Middle Eastern cuisines, it is a staple for dishes like biryani, pilaf, and curries.",
    'Ipsala': "Ipsala rice originates from the Ipsala region in Turkey and is known for its medium-grain size and firm texture. It is frequently used in traditional Turkish pilafs and casseroles, adding a distinct regional touch to meals.",
    'Jasmine': "Jasmine rice is a fragrant long-grain variety popular in Southeast Asian cuisine. Its soft and slightly sticky texture makes it perfect for dishes like Thai curries, stir-fries, and fried rice.",
    'Karacadag': "Karacadag rice is grown in the southeastern regions of Turkey and is prized for its unique taste and firm texture. It is a staple ingredient in many regional Turkish dishes and pairs well with hearty stews."
}

# Function to preprocess and predict using a model
def predict_image(model, img, img_size):
    img = img.convert("RGB")  # Ensure the image is in RGB format
    img = img.resize(img_size)  # Resize image
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return prediction

# Streamlit Sidebar Navigation
st.sidebar.title("RIC")
navigation = st.sidebar.radio("Go to", ["Home", "Rice Image Classification","About", "Team Members"])

# Display content based on selected page
if navigation == "Home":
    st.title("Welcome to the Rice Variety Classification Project")
    st.markdown("""
    This project classifies rice varieties using advanced deep learning models. 
    It focuses on rice commonly grown in Turkey and provides a detailed description of each variety.
    
    Rice classification is significant for understanding different varieties' usage and quality, 
    aiding farmers, food experts, and the culinary world.
    """)

elif navigation == "Rice Image Classification":
    st.title("Classify Rice Varieties")
    st.markdown("""
    Upload an image of rice to classify its variety. This tool uses advanced machine learning models 
    like ResNet, VGG16, and CNN to analyze the image and predict the type of rice.

    Each variety has unique characteristics, including size, shape, texture, and aroma, which are 
    considered during classification. Once classified, you can learn more about the predicted variety 
    and its culinary applications.
    """)

    uploaded_file = st.file_uploader("Upload a rice image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        model_option = st.selectbox("Choose a model", ["ResNet", "VGG16", "CNN"])
        if model_option == "ResNet":
            prediction = predict_image(resnet_model, img, resnet_img_size)
        elif model_option == "VGG16":
            prediction = predict_image(vgg16_model, img, vgg16_img_size)
        elif model_option == "CNN":
            prediction = predict_image(cnn_model, img, cnn_img_size)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Display predicted class as heading
        st.subheader(f"Predicted Rice Variety: {predicted_class}")
        st.write(f"**Confidence:** {confidence:.2f}%")
        st.write(f"{descriptions.get(predicted_class, 'No description available.')}")
    


elif navigation == "Team Members":
    st.title("Meet the Team")
    st.write("### Team Members:")
    st.write("- Bhavya Sri Mulakala")
    st.write("- Pavani Bayya")
    st.write("- Joshmika Bonu")
    st.write("- Rajeswari Mogili")
    st.write("### Project Guide:")
    st.write("**Abdul Aziz Sir**")

elif navigation == "About":
    st.title("About the Project")
    st.markdown("""
    ### Welcome to the Rice Variety Classification Project

    This project is designed to classify various rice varieties using advanced deep learning techniques. 
    By analyzing the physical characteristics of rice grains, such as size, shape, and texture, the system can predict the specific variety of rice. 
    It focuses on rice varieties commonly grown in Turkey, a country known for its rich agricultural heritage and unique regional rice types.

    #### Key Objectives:
    1. **Precision in Classification:** 
       Leverage state-of-the-art deep learning models (ResNet, VGG16, and CNN) to achieve accurate classification of rice varieties.
    2. **Culinary and Agricultural Insights:** 
       Provide detailed descriptions of each rice type, their culinary applications, and how they are used in traditional Turkish and global cuisines.
    3. **Support for Agriculture and Food Industry:** 
       Assist farmers, food technologists, and culinary professionals in identifying and understanding the characteristics of different rice varieties, aiding in quality control and selection.

    #### Significance of the Project
    Rice is a staple food in many countries, and its classification plays a critical role in ensuring the right variety is used for specific dishes. 
    The project's emphasis on rice varieties from Turkey also highlights the country's unique agricultural products, making it an invaluable tool for:
    - Farmers seeking to improve crop quality.
    - Chefs and food enthusiasts exploring the best varieties for their recipes.
    - Researchers studying the characteristics and origins of different rice types.

    Through this tool, users can not only classify rice but also learn about its origin, flavor profile, and ideal culinary uses.
    """)


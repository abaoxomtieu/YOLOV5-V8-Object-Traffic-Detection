import streamlit as st
import requests
from PIL import Image
from io import BytesIO

# Define the FastAPI server URL
fastapi_url = "http://127.0.0.1:8000"  # Update the URL to match your FastAPI server URL

st.title("FastAPI Image Prediction Demo")

# Upload an image for prediction
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Make a prediction request to the FastAPI server
    if st.button("Predict"):
        try:
            # Send the image to the FastAPI server for prediction
            response = requests.post(fastapi_url + "/predict/", files={"file": uploaded_image})
            response.raise_for_status()

            # Display the prediction result
            result_image = Image.open(BytesIO(response.content))
            st.image(result_image, caption="Prediction Result", use_column_width=True)

        except requests.exceptions.RequestException as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Upload an image for prediction.")

st.write("Note: Make sure your FastAPI server is running and accessible at the specified URL.")

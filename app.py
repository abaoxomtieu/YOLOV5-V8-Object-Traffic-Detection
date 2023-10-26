import streamlit as st
import requests
from io import BytesIO
from PIL import Image

# Set the FastAPI service URL
api_url = "http://127.0.0.1:8000"  # Replace with the URL of your FastAPI service

st.title("Image Prediction and Real-Time Object Detection")

# Use a radio button to choose the service
selected_service = st.radio("Select a service", ["Image Prediction", "Real-Time Object Detection"])

if selected_service == "Image Prediction":
    # Upload an image for prediction
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Make a prediction request to the FastAPI server
        if st.button("Predict"):
            try:
                # Send the image to the FastAPI server for prediction
                response = requests.post(api_url + "/predict/", files={"file": uploaded_image})
                response.raise_for_status()

                # Display the prediction result
                result_image = Image.open(BytesIO(response.content))
                st.image(result_image, caption="Prediction Result", use_column_width=True)

            except requests.exceptions.RequestException as e:
                st.error(f"An error occurred: {e}")
    else:
        st.info("Upload an image for prediction.")
else:
    # Stream the video feed from the FastAPI service
    iframe = f'<iframe src="{api_url}" width="1280" height="720"></iframe>'
    st.markdown(iframe, unsafe_allow_html=True)

st.write("Note: Make sure your FastAPI server is running and accessible at the specified URL.")

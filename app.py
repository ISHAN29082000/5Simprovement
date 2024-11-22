import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Initialize YOLO model
model = YOLO('yolov5su.pt')  # Ensure this matches the file you downloaded

# Streamlit app title
st.title("Real-time Object Detection Scanner")
st.subheader("Upload or Take a Picture to Detect Objects")

# Instructions for the user
st.markdown("""
    - 📸 Use the **Upload Image** button to upload a photo taken from your mobile camera.
    - 🚀 Once uploaded, YOLOv5 will detect and highlight objects in the image.
""")

# Image upload (mobile-friendly)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    img = Image.open(uploaded_file)
    
    # Convert to numpy array for YOLO processing
    img_np = np.array(img)

    # Run YOLO model to detect objects
    results = model(img_np)

    # Render bounding boxes and labels on the image
    result_img = results[0].plot()  # Annotated image with bounding boxes

    # Display the result
    st.image(result_img, caption="Detected Objects", use_column_width=True)

    # Optionally, save the result if needed
    if st.button("Save Image"):
        result_img_pil = Image.fromarray(result_img)
        result_img_pil.save("detected_image.jpg")
        st.success("Image saved successfully!")


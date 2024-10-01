import streamlit as st
from PIL import Image
from ultralytics import YOLO
import pandas



st.set_page_config(
    page_title = "Dental Disease Recognition",
    layout = "wide"
    )




model_src = r"./model.pt"

image = None
detected_classes = []

image = st.file_uploader("Upload an oral image:", type=["jpg", "png", "jpeg"])

if image is not None:
    img = Image.open(image)
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image(img, caption="Input Image")
    
    detect_button = st.button ("DETECT IMAGE", use_container_width=True)
    
    if detect_button :
        model = YOLO(model_src)        
        res = model.predict(img,save=True)
        box = res[0].boxes.xyxy.tolist()
        res_plotted = res[0].plot()[:, :, ::-1]
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image(res_plotted, caption='Output Image')

else:
    st.info("Please upload an oral image to continue the detection task")


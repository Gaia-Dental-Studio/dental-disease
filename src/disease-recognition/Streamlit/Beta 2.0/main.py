import streamlit as st
from PIL import Image, ImageDraw
from ultralytics import YOLO
import pandas
import numpy
import cv2



st.set_page_config(
    page_title = "Dental Disease Recognition",
    layout = "wide"
    )



#model path
disease = r".\model\disease.pt"
upper = r".\model\upper.pt"
lower = r".\model\lower.pt"
front = r".\model\front.pt"


def get_tooth_number_model(orientation):
  """Selects the appropriate tooth numbering model based on orientation."""
  if orientation == "Upper":
    return YOLO(upper)
  elif orientation == "Lower":
    return YOLO(lower)
  elif orientation == "Front":
    return YOLO(front)
  else:
     st.info("Please choose the orientation of oral image")
     


image = None
detected_classes = []


orientation = st.radio("Image Orientation", ("Upper", "Lower", "Front"))

image = st.file_uploader("Upload an oral image:", type=["jpg", "png", "jpeg"])

if image is not None:
  img = Image.open(image)
  imgs = numpy.array(img)
  left_co, cent_co, last_co = st.columns(3)
  with cent_co:
    st.image(img, caption="Input Image")

  detect_button = st.button("DETECT IMAGE", use_container_width=True)

  if detect_button:
    # Load disease detection model
    disease_model = YOLO(disease)

    # Load tooth numbering model based on orientation
    tooth_model = get_tooth_number_model(orientation)

    # Disease and tooth number detection
    disease_res = disease_model.predict(img)
    tooth_number = tooth_model(img)
    
    
    for r in disease_res:
        disease_classes = disease_model.names
        
        for box in r.boxes:
            if box.conf[0] > 0.4 :
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                                
                cv2.rectangle(imgs, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                cv2.putText(imgs, f'{disease_classes[int(box.cls[0])]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    for r in tooth_number :
        tooth_numbers = tooth_model.names
        
        for box in r.boxes :
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cls = int(box.cls[0])
                
                cv2.putText(imgs, f'{tooth_numbers[int(box.cls[0])]}', (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Display results
    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
      st.image(imgs, caption="Output Image")

else:
    st.info("Please upload an oral image to continue the detection task")


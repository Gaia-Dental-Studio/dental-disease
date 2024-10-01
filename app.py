# Python In-built packages
from pathlib import Path
import PIL
import numpy as np
from ultralytics.utils.ops import scale_image
from sklearn.cluster import KMeans

# External packages
import streamlit as st

# Local Modules
import settings
import helper
import cv2
# Setting page layout
st.set_page_config(
    page_title="DentalMate",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main page heading
st.title("Dentalmate")

st.divider()
st.header("FDI Teeth Notation")

st.image('./assets/numbering_syd.gif',"FDI Notation",use_column_width=True, width=10)
# Sidebar
st.sidebar.header("ML Model Config")

# Model Options
model_type = st.sidebar.radio(
    "Select Task", ['Detection', 'Segmentation'])

confidence = float(st.sidebar.slider(
    "Select Model Confidence", 25, 100, 30)) / 100

# Selecting Detection Or Segmentation
if model_type == 'Detection':
    model_path = Path(settings.DETECTION_MODEL)
elif model_type == 'Segmentation':
    model_path = Path(settings.SEGMENTATION_MODEL)

# Load Pre-trained ML Model
try:
    model_det = helper.load_model(Path(settings.DETECTION_MODEL))
    model_seg = helper.load_model(Path(settings.SEGMENTATION_MODEL))
    model_dis_seg = helper.load_model(Path(settings.DISEASE_SEGMENTATION_MODEL))
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

st.sidebar.header("Image Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

st.sidebar.header("Jaw Image")
jaw_select = st.sidebar.radio("Select Jaw Position",['Upper','Lower','Front'])
source_img = None
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2 = st.columns(2)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default Image",
                         use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         use_column_width=True)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
       
                if model_type=='Detection':
                    res = model_det.predict(uploaded_image,
                                    conf=confidence
                                    )
                # print(res[0])
                # res[0].names={0: 'abscess', 1: 'cervical caries', 2: 'dentine caries', 3: 'enamel caries', 4: 'interproximal caries', 5: 'deep caries', 6: 'radicular caries', 7: 'rampant caries', 8: 'tartar', 9: 'ulcer'}
       
       
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image',
                            use_column_width=True)
                else:
                    boxes, masks, cls, probs = helper.predict_on_image(model_seg, uploaded_image)
                    image_with_masks = np.copy(uploaded_image)
                    for mask_i in masks:
                        image_with_masks = helper.overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0)
                    res = model_det.predict(uploaded_image,conf=confidence)
                    
                    centroids=[]
                    results = model_seg(uploaded_image)
                    for i in results[0].masks.data:
                        mask = (i.numpy()*255).astype('uint8')
                        mask = scale_image(mask, results[0].masks.orig_shape)
                        centroid = np.mean(np.argwhere(mask),axis=0)
                        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                        centroids.append([centroid_y,centroid_x,mask])
                    centroids = sorted(centroids,key= lambda x:x[1])
                    rotated_data = [[point[1], point[0]] for point in centroids]
                    teeth_data_front = {}
                    teeth_data={}
                    front_teeth ={}

                    ##

                    res_segment = model_dis_seg(uploaded_image,conf=confidence)

                    for r in res_segment:
                        rboxes = r.boxes  # Boxes object for bbox outputs
                        rmasks = r.masks  # Masks object for segment masks outputs
                        rprobs = r.probs  # Class probabilities for classification outputs

                    if rmasks is not None:
                        rmasks = rmasks.data.cpu()
                        for seg, box in zip(rmasks.data.cpu().numpy(), rboxes):

                            seg = cv2.resize(seg, (image_with_masks.shape[0:2][::-1]))
                            # print(seg)
                            image_with_masks = helper.overlay(image_with_masks, seg, (0,255,0), 0.4)
                            # print(box.cls)
                            xmin = int(box.data[0][0])
                            ymin = int(box.data[0][1])
                            xmax = int(box.data[0][2])
                            ymax = int(box.data[0][3])
                            
                            helper.plot_one_box([xmin, ymin, xmax, ymax], image_with_masks, (0,0,0), f'{res_segment[0].names[int(box.cls)]} {float(box.conf):.3}')

                    # centroid_disease=[]

                    # for mask, cls, conf in zip(res_segment[0].masks.data, res_segment[0].boxes.cls.tolist(), res_segment[0].boxes.conf.tolist()):
                    #     confidence = conf
                    #     detected_class = cls
                    #     name = res_segment[0].names[int(cls)]
                    #     mask = (mask.numpy()*255).astype('uint8')
                    #     mask = scale_image(mask, results[0].masks.orig_shape)
                    #     centroid = np.mean(np.argwhere(mask),axis=0)
                    #     centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                    #     centroid_disease.append([[centroid_y,centroid_x],name,confidence])



# Assign unique names to points in the left and right quadrants
                    if jaw_select=='Lower':
                        left_quadrant_names = [f'{i+31}' for i in range(len(centroids)//2)][::-1]
                        right_quadrant_names = [f'{i+41}' for i in range(len(centroids))]
                    elif jaw_select=="Upper":
                        left_quadrant_names = [f'{i+11}' for i in range(len(centroids)//2)][::-1]
                        right_quadrant_names = [f'{i+21}' for i in range(len(centroids))]
                    elif jaw_select=="Front":
                        y_min_values = []
                        for i in results[0].masks.data:
                            mask = (i.numpy() * 255).astype('uint8')
                            mask = scale_image(mask, results[0].masks.orig_shape)
                            
                            # Get the coordinates of non-zero values in the mask
                            coordinates = np.argwhere(mask)
                            
                            if coordinates.size > 0:
                                # Extract the y-coordinates and find the minimum
                                min_y = int(np.min(coordinates[:, 0]))
                                y_min_values.append(min_y)
                            else:
                                # Handle the case when the mask has no non-zero values
                                y_min_values.append(None)  # You can choose a default value or handle it accordingly
                        coordinates_list = []
                        teeth_id = 0

                        for i in results[0].masks.data:
                            mask = (i.numpy() * 255).astype('uint8')
                            mask = scale_image(mask, results[0].masks.orig_shape)
                            
                            # Get the coordinates of non-zero values in the mask
                            coordinates = np.argwhere(mask)
                            
                            if coordinates.size > 0:
                                # Find the index of the first occurrence of the minimum y-coordinate
                                min_y_index = np.argmin(coordinates[:, 0])
                                # Extract the minimum y-coordinate and corresponding x-coordinate
                                min_y, min_x = int(coordinates[min_y_index, 0]), int(coordinates[min_y_index, 1])
                                coordinates_list.append([min_y, min_x])
                                teeth_data_front[teeth_id] = {"coordinates":coordinates,"top_part":[min_y,min_x],"mask":mask}
                                teeth_id+=1
                            
                            else:
                                # Handle the case when the mask has no non-zero values
                                coordinates_list.append(None)  # You can choose a default value or handle it accordingly
                        for idx, obj in teeth_data_front.items():
                            centr = np.mean(np.argwhere(obj['mask']),axis=0)
                            obj['centroids'] = [(int(centr[1])),(int(centr[0]))]
                        top_part_values = [obj['top_part'][0] for obj in teeth_data_front.values()]
                        X = np.array(top_part_values)
                        X = X.reshape(-1,1)

                        # Specify the number of clusters (k)
                        k = 2

                        # Perform k-means clustering
                        kmeans = KMeans(n_clusters=k, random_state=42,tol=1e-5)
                        kmeans.fit(X)

                        # Get the cluster labels
                        cluster_labels = kmeans.labels_
                        for idx, obj in teeth_data_front.items():
                            obj['cluster'] = cluster_labels[idx]

                        for cluster_id in range(2):
                            cluster_indices = [idx for idx, obj in teeth_data_front.items() if obj['cluster'] == cluster_id]
                            cluster_data = [(idx, obj) for idx, obj in teeth_data_front.items() if obj['cluster'] == cluster_id]
                            # print(cluster_indices)
                            # print(cluster_data)
                            # Sort by the x-axis (assuming x-axis is the first element of 'top_part')
                            sorted_cluster_data = sorted(cluster_data, key=lambda x: x[1]['top_part'][1])
                            half_len = len(sorted_cluster_data) // 2
                            print(half_len)
                            if cluster_id ==0:
                                left_range = list(range(11, half_len + 11))[::-1]  # Reverse the left range
                                right_range = list(range(21, len(sorted_cluster_data) + 21))
                            else:
                                left_range = list(range(41, half_len + 41))[::-1]  # Reverse the left range
                                right_range = list(range(31, len(sorted_cluster_data) + 31))
                            full_range = left_range+right_range
                            for i, (idx, obj) in enumerate(sorted_cluster_data):
                                # print(i)
                                if i <= half_len:
                                    obj['sorted_position'] = full_range[i]
                                else:
                                    obj['sorted_position'] = full_range[i]
                                front_teeth[idx]  = obj
                        

                    if jaw_select=='Upper' or jaw_select=="Lower":

                        x_coordinates = [point[0] for point in rotated_data]
                        y_coordinates = [-point[1] for point in rotated_data]
                        dental_namings = left_quadrant_names+right_quadrant_names

                        for i in range(len(masks)):
                            cv2.putText(image_with_masks, dental_namings[i], (centroids[i][1],centroids[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                            teeth_data[dental_namings[i]]={'centroid':(centroids[i][1],centroids[i][0])}
                        
                        for j in teeth_data.keys():
                            for i in results[0].masks.data:
                                mask = (i.numpy()*255).astype('uint8')
                                mask = scale_image(mask, results[0].masks.orig_shape)
                                centroid = np.mean(np.argwhere(mask),axis=0)
                                centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                                if centroid_x == teeth_data[j]['centroid'][0] and centroid_y ==teeth_data[j]['centroid'][1]:
                                    teeth_data[j]['mask']= mask
                                    break
                        
                        for i in teeth_data.keys():
                            teeth_data[i]['teeth_disease']= []
                            teeth_data[i]['teeth_disease_coord'] = []
                            teeth_data[i]['teeth_disease_seg']=[]
                            teeth_data[i]['teeth_disease_seg_coord']=[]
                            teeth_data[i]['teeth_disease_seg_conf']=[]
                            teeth_data[i]['mask_shape']=teeth_data[i]['mask'].shape
                        
                        for detection in res[0].boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = detection
                            vehicle_bounding_boxes=[]
                            vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                            center_dis = [int((x1+x2)//2),int((y1+y2)//2)]
                            # print(center_dis)
                            for i in teeth_data.keys():
                                if teeth_data[i]['mask'][center_dis[1]][center_dis[0]][0]==255:
                                    # print("haer")
                                    teeth_data[i]['teeth_disease'].append(res[0].names[class_id])
                                    teeth_data[i]['teeth_disease_coord'].append(center_dis)

                        if res_segment[0].masks != None:

                            for mask_, cls_, conf_,boxes_ in zip(res_segment[0].masks.data, res_segment[0].boxes.cls.tolist(), res_segment[0].boxes.conf.tolist(),res_segment[0].boxes.xyxy.tolist()):
                                confidence_dis = conf_
                                detected_class = cls_
                                name = res_segment[0].names[int(cls_)]
                                mask = (mask_.numpy()*255).astype('uint8')
                                mask = scale_image(mask, res_segment[0].masks.orig_shape)
                                centroid = np.mean(np.argwhere(mask),axis=0)
                                centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                                # centroid_disease.append([[centroid_y,centroid_x],name,confidence])
                                
                                for i in teeth_data.keys():
                                    if teeth_data[i]['mask'][centroid_y][centroid_x][0]==255:
                                        teeth_data[i]['teeth_disease_seg'].append(name)
                                        teeth_data[i]['teeth_disease_seg_coord'].append([centroid_y,centroid_x])
                                        teeth_data[i]['teeth_disease_seg_conf'].append(confidence_dis)
                    
                    else:
                        for idx,obj in front_teeth.items():
                            obj['teeth_disease']= []
                            obj['teeth_disease_coord'] = []
                            obj['teeth_disease_seg']=[]
                            obj['teeth_disease_seg_coord']=[]
                            obj['teeth_disease_seg_conf']=[]
 
                        for detection in res[0].boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = detection
                            vehicle_bounding_boxes=[]
                            vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                            center_dis = [int((x1+x2)//2),int((y1+y2)//2)]
                            # print(res[0].names)
                            for idx,obj in front_teeth.items():
                                if obj['mask'][center_dis[1]][center_dis[0]][0]==255:
                                    obj['teeth_disease'].append(res[0].names[class_id])
                                    obj['teeth_disease_coord'].append(center_dis)

                        if res_segment[0].masks != None:

                            for mask_, cls_, conf_ in zip(res_segment[0].masks.data, res_segment[0].boxes.cls.tolist(), res_segment[0].boxes.conf.tolist()):
                                confidence_dis = conf_
                                detected_class = cls_
                                name = res_segment[0].names[int(cls_)]
                                mask = (mask_.numpy()*255).astype('uint8')
                                mask = scale_image(mask, res_segment[0].masks.orig_shape)
                                centroid = np.mean(np.argwhere(mask),axis=0)
                                centroid_x, centroid_y = int(centroid[1]), int(centroid[0])
                                for idx,obj in front_teeth.items():
                                    if obj['mask'][centroid_y][centroid_x][0]==255:
                                        obj['teeth_disease_seg'].append(name)
                                        obj['teeth_disease_seg_coord'].append([centroid_y,centroid_x])
                                        obj['teeth_disease_seg_conf'].append(confidence_dis)

                        for idx,obj in front_teeth.items():
                            cv2.putText(image_with_masks, str(obj['sorted_position']), (obj['centroids'][0],obj['centroids'][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                            # teeth_data[dental_namings[i]]={'centroid':(centroids[i][1],centroids[i][0])}
      
                    for detection in res[0].boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = detection
                        vehicle_bounding_boxes=[]
                        vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                        for bbox in vehicle_bounding_boxes:
                            # print("SKD")
                            cv2.rectangle(image_with_masks, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                            cv2.putText(image_with_masks, res[0].names[class_id], [int(x1),int(y1)+30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    
                    st.image(image_with_masks,caption='Detected Image',
                                            use_column_width=True)   
                try:
                    with st.expander("Detection Results"):
                        if model_type=='Detection':
                            for detection in res[0].boxes.data.tolist():
                                x1, y1, x2, y2, score, class_id = detection
                                st.write(res[0].names[class_id] )
                                st.write("\nConfidence :"+str(score))
                                st.divider()
                        else:
                            if jaw_select=='Upper' or jaw_select=="Lower":

                                st.write("Number teeth detected: "+ str(len(centroids)))
                                st.divider()
                                for i in teeth_data.keys():
                                    if teeth_data[i]['teeth_disease']!=[]:
                                        st.write("Teeth Number : " + str(i))
                                        teeth_dis = []
                                        teeth_treat= []
                                        for j in teeth_data[i]['teeth_disease']:
                                            # st.write(j)
                                            teeth_dis_ = j.split('-')
                                            # print(teeth_dis_)
                                            teeth_dis.append(teeth_dis_[0]) 
                                            teeth_treat.append(teeth_dis_[1]) 
                                            st.write("Teeth Disease: " + str(teeth_dis_[0]))
                                            st.write("Teeth Treatment: " + str(teeth_dis_[1]))
                                            # st.write("Teeth Disease: " + str(*teeth_data[i]['teeth_disease']))
                                        st.divider()
                                    if teeth_data[i]['teeth_disease_seg']!=[]:
                                        st.write("Teeth Number : " + str(i))

                                        st.write("Teeth Disease Segmentation Model: "+str(teeth_data[i]['teeth_disease_seg']))

                                print(teeth_data)
                            else:
                                st.write("Number teeth detected: "+ str(len(front_teeth.keys())))
                                st.divider()
                                for idx,obj in front_teeth.items():
                                    if obj['teeth_disease']!=[]:
                                        st.write("Teeth Number : " + str(obj['sorted_position']))
                                        teeth_dis = []
                                        teeth_treat= []
                                        for j in obj['teeth_disease']:
                                            # st.write(j)
                                            teeth_dis_ = j.split('-')
                                            # print(teeth_dis_)
                                            teeth_dis.append(teeth_dis_[0]) 
                                            teeth_treat.append(teeth_dis_[1]) 
                                            st.write("Teeth Disease Detection: " + str(teeth_dis_[0]))
                                            st.write("Teeth Treatment: " + str(teeth_dis_[1]))
                                            # st.write("Teeth Disease: " + str(*teeth_data[i]['teeth_disease']))
                                        st.divider()
                                    if obj['teeth_disease_seg']!=[]:
                                        st.write("Teeth Number : " + str(obj['sorted_position']))
                                        teeth_dis = []
                                        teeth_treat= []
                                        st.write("Teeth Disease Segmentation: "+ str(obj['teeth_disease_seg']))
                                        st.divider()
                                print(front_teeth)

                        # for box in boxes:
                        #     st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")



else:
    st.error("Please select a valid source type!")

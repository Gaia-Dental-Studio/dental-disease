from flask import Flask, request, jsonify,send_from_directory
from PIL import Image
import io
import numpy as np
import helper
from pathlib import Path
import settings
from ultralytics.utils.ops import scale_image
import json

import cv2

import io
import time
import base64

from sklearn.cluster import KMeans


app = Flask(__name__,static_url_path='/output')

# Load Model

model_det = helper.load_model(Path(settings.DETECTION_MODEL))
model_seg = helper.load_model(Path(settings.SEGMENTATION_MODEL))
model_dis_seg = helper.load_model(Path(settings.DISEASE_SEGMENTATION_MODEL))


# Send Image File
@app.route('/output/<filename>')
def output_image(filename):
    # Define the directory where your images are stored
    image_directory = './output'  # Update this with the actual directory path
    return send_from_directory(image_directory, filename)

# Front Detection

@app.route('/upload-lower', methods=['POST'])
def upload_lower():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Read image file
            uploaded_image = Image.open(io.BytesIO(file.read()))
            
          
            width, height = uploaded_image.size
            
            boxes, masks, cls, probs = helper.predict_on_image(model_seg, uploaded_image)
            image_with_masks = np.copy(uploaded_image)
            for mask_i in masks:
                image_with_masks = helper.overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0)
            res = model_det.predict(uploaded_image,conf=.2)
            
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

            res_segment = model_dis_seg(uploaded_image,conf=0.2)

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
            left_quadrant_names = [f'{i+31}' for i in range(len(centroids)//2)][::-1]
            right_quadrant_names = [f'{i+41}' for i in range(len(centroids))]
            # elif jaw_select=="Upper":
            #     left_quadrant_names = [f'{i+11}' for i in range(len(centroids)//2)][::-1]
            #     right_quadrant_names = [f'{i+21}' for i in range(len(centroids))]    


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
            
            another_disease = {}
            another_disease_id=0

            for detection in res[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                vehicle_bounding_boxes=[]
                vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                center_dis = [int((x1+x2)//2),int((y1+y2)//2)]
                flag = False
                # print(center_dis)
                for i in teeth_data.keys():
                    if teeth_data[i]['mask'][center_dis[1]][center_dis[0]][0]==255:
                        # print("haer")
                        teeth_data[i]['teeth_disease'].append(res[0].names[class_id])
                        teeth_data[i]['teeth_disease_coord'].append(center_dis)
                        flag = True
                        break
                if not flag:
                    another_disease[another_disease_id]= {"name": res[0].names[class_id],"center_dis":center_dis}
                    another_disease_id+=1

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
                    flag= False
                    for i in teeth_data.keys():
                        if teeth_data[i]['mask'][centroid_y][centroid_x][0]==255:
                            teeth_data[i]['teeth_disease_seg'].append(name)
                            teeth_data[i]['teeth_disease_seg_coord'].append([centroid_y,centroid_x])
                            teeth_data[i]['teeth_disease_seg_conf'].append(confidence_dis)
                            flag=True
                            break
                    if not flag :
                        another_disease[another_disease_id] = {"name":name,"center_dis": [centroid_x,centroid_y]}
            for detection in res[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                vehicle_bounding_boxes=[]
                vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                for bbox in vehicle_bounding_boxes:
                    # print("SKD")
                    cv2.rectangle(image_with_masks, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                    cv2.putText(image_with_masks, res[0].names[class_id], [int(x1),int(y1)+30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

            image_output = Image.fromarray(image_with_masks)
            image_name = f'{int(time.time())}.png'

            image_output.save('./output/'+image_name)
            for i in teeth_data.keys():
                teeth_data[i]['mask']=[]
            print(teeth_data)
            # Return JSON response
            return jsonify({
                'image' : request.host_url + 'output/' + image_name,
                'teeth_data': teeth_data,
                'another_disease': another_disease,
                'image_width':width,
                'image_height': height

            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Unknown error'})


@app.route('/upload-upper', methods=['POST'])
def upload_upper():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Read image file
            uploaded_image = Image.open(io.BytesIO(file.read()))
            
          
            width, height = uploaded_image.size
            
            boxes, masks, cls, probs = helper.predict_on_image(model_seg, uploaded_image)
            image_with_masks = np.copy(uploaded_image)
            for mask_i in masks:
                image_with_masks = helper.overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0)
            res = model_det.predict(uploaded_image,conf=.2)
            
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

            res_segment = model_dis_seg(uploaded_image,conf=0.2)

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

         
            left_quadrant_names = [f'{i+11}' for i in range(len(centroids)//2)][::-1]
            right_quadrant_names = [f'{i+21}' for i in range(len(centroids))]    


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
            
            another_disease = {}
            another_disease_id=0

            for detection in res[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                vehicle_bounding_boxes=[]
                vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                center_dis = [int((x1+x2)//2),int((y1+y2)//2)]
                flag = False
                # print(center_dis)
                for i in teeth_data.keys():
                    if teeth_data[i]['mask'][center_dis[1]][center_dis[0]][0]==255:
                        # print("haer")
                        teeth_data[i]['teeth_disease'].append(res[0].names[class_id])
                        teeth_data[i]['teeth_disease_coord'].append(center_dis)
                        flag = True
                        break
                if not flag:
                    another_disease[another_disease_id]= {"name": res[0].names[class_id],"center_dis":center_dis}
                    another_disease_id+=1

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
                    flag= False
                    for i in teeth_data.keys():
                        if teeth_data[i]['mask'][centroid_y][centroid_x][0]==255:
                            teeth_data[i]['teeth_disease_seg'].append(name)
                            teeth_data[i]['teeth_disease_seg_coord'].append([centroid_y,centroid_x])
                            teeth_data[i]['teeth_disease_seg_conf'].append(confidence_dis)
                            flag=True
                            break
                    if not flag :
                        another_disease[another_disease_id] = {"name":name,"center_dis": [centroid_x,centroid_y]}
            for detection in res[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                vehicle_bounding_boxes=[]
                vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                for bbox in vehicle_bounding_boxes:
                    # print("SKD")
                    cv2.rectangle(image_with_masks, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                    cv2.putText(image_with_masks, res[0].names[class_id], [int(x1),int(y1)+30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            image_output = Image.fromarray(image_with_masks)
            
            image_name = f'{int(time.time())}.png'

            image_output.save('./output/'+image_name)
            for i in teeth_data.keys():
                teeth_data[i]['mask']=[]
            print(teeth_data)
            # Return JSON response
            return jsonify({
                'image': request.host_url + 'output/' + image_name,
                'teeth_data': teeth_data,
                'another_disease': another_disease,
                'image_width':width,
                'image_height': height

            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Unknown error'})

@app.route('/upload-front', methods=['POST'])
def upload_front():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            # Read image file
            uploaded_image = Image.open(io.BytesIO(file.read()))
            
          
            width, height = uploaded_image.size
            
            boxes, masks, cls, probs = helper.predict_on_image(model_seg, uploaded_image)
            image_with_masks = np.copy(uploaded_image)
            for mask_i in masks:
                image_with_masks = helper.overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0)
            res = model_det.predict(uploaded_image,conf=.2)
            
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

            res_segment = model_dis_seg(uploaded_image,conf=0.2)

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
                obj['cluster'] = int(cluster_labels[idx])

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
            for idx,obj in front_teeth.items():
                obj['teeth_disease']= []
                obj['teeth_disease_coord'] = []
                obj['teeth_disease_seg']=[]
                obj['teeth_disease_seg_coord']=[]
                obj['teeth_disease_seg_conf']=[]

            for detection in res[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                vehicle_bounding_boxes=[]
                vehicle_bounding_boxes.append([int(x1), int(y1), int(x2), int(y2),class_id])
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
            for detection in res[0].boxes.data.tolist():
                x1, y1, x2, y2, score, class_id = detection
                vehicle_bounding_boxes=[]
                vehicle_bounding_boxes.append([x1, y1, x2, y2,class_id])
                for bbox in vehicle_bounding_boxes:
                    # print("SKD")
                    cv2.rectangle(image_with_masks, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 3)
                    cv2.putText(image_with_masks, res[0].names[class_id], [int(x1),int(y1)+30], cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            image_output = Image.fromarray(image_with_masks)
            image_name = f'{int(time.time())}.png'
            
            image_output.save('./output/'+image_name)

            for i in front_teeth.keys():
                front_teeth[i]['mask']=[]
                front_teeth[i]['coordinates']=[]
            print(front_teeth)

# Return JSON response
            return jsonify({
                'image' : request.host_url + 'output/' + image_name,
                'teeth_data': front_teeth,
                'image_width':width,
                'image_height': height

            })
        except Exception as e:
            return jsonify({'error': str(e)})
    
    return jsonify({'error': 'Unknown error'})
    

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)  # debug=True causes Restarting with stat

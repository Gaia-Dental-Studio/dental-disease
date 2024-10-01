from ultralytics.utils.ops import scale_image
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import pandas as pd


def showing_masking(image_path):
    img =cv2.imread(image_path)
    boxes, masks, cls, probs = predict_on_image(model, img)

    # overlay masks on original image
    image_with_masks = np.copy(img)
    for mask_i in masks:
        image_with_masks = overlay(image_with_masks, mask_i, color=(0,255,0), alpha=0.3)

    return plt.imshow(cv2.cvtColor(image_with_masks, cv2.COLOR_BGR2RGB))


def predict_on_image(model, img):
    result = model(img)[0]

    # detection
    # result.boxes.xyxy   # box with xyxy format, (N, 4)
    cls = result.boxes.cls.cpu().numpy()    # cls, (N, 1)
    probs = result.boxes.conf.cpu().numpy()  # confidence score, (N, 1)
    boxes = result.boxes.xyxy.cpu().numpy()   # box with xyxy format, (N, 4)

    # segmentation
    masks = result.masks.data.cpu().numpy()     # masks, (N, H, W)
    print(masks.shape)
    masks = np.moveaxis(masks, 0, -1) # masks, (H, W, N)
    # rescale masks to original image
    # masks = scale_image(masks.shape[:2], masks, result.masks.orig_shape)
    masks = scale_image(masks, result.masks.orig_shape)
    print(masks.shape)
    

    masks = np.moveaxis(masks, -1, 0) # masks, (N, H, W)

    return boxes, masks, cls, probs


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Params:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (1024, 1024))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined


def normalize_points(points):
    """
    Normalize the points to the range [0,1].

    :param points: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the points.
    :return: A list of tuples representing the normalized points.
    """
    x_coords, y_coords = zip(*points)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    normalized_points = [((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min)) for x, y in points]
    return normalized_points


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points in 2D space.

    :param point1: A tuple (x1, y1) representing the first point.
    :param point2: A tuple (x2, y2) representing the second point.
    :return: The Euclidean distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def plot_points_and_lines_with_distances(points):
    """
    Plot multiple points and lines connecting them in 2D space, showing distances between points.
    Save the distances in a DataFrame.

    :param points: A list of tuples [(x1, y1), (x2, y2), ..., (xn, yn)] representing the points.
    """
    # Unzip the list of points into two lists: x coordinates and y coordinates
    x_coords, y_coords = zip(*points)

    # Create a new figure
    plt.figure()

    # Plot the points
    plt.scatter(x_coords, y_coords, color='r')

    # Plot the lines connecting consecutive points and annotate distances
    distances = []
    for i in range(len(points) - 1):
        point1 = points[i]
        point2 = points[i + 1]
        distance = calculate_distance(point1, point2)
        distances.append(distance)
        
        # Plot the line
        plt.plot([point1[0], point2[0]], [point1[1], point2[1]], color='b')
        
        # Annotate the distance
        mid_x = (point1[0] + point2[0]) / 2
        mid_y = (point1[1] + point2[1]) / 2
        plt.text(mid_x, mid_y, f'{distance:.2f}', fontsize=9, ha='center')

    # Set labels
    plt.xlabel('X axis')
    plt.ylabel('Y axis')

    # Show the plot
    plt.show()

    # Save the distances in a DataFrame
    df = pd.DataFrame(distances, columns=['Distance'])
    print(df)
    return df, distances

# # Function to generate synthetic distances between teeth
# def generate_tooth_distances(num_teeth):
#     distances = []
#     for i in range(num_teeth - 1):
#         distance = round(random.uniform(0.1, 2.0), 2)  # Random distance between 0.1 and 2.0
#         distances.append(distance)
#     return distances


# Function to detect loose teeth based on abnormal distances
def detect_loose_teeth(distances, threshold_factor=1.5):
    avg_distance = sum(distances) / len(distances)
    loose_teeth_indices = []

    new_distance = []
    
    for i, distance in enumerate(distances):
        if distance > avg_distance * threshold_factor:
            num_tooth_lose = round(distance/avg_distance)
            for j in range(num_tooth_lose):
                new_distance.append(distance/avg_distance)
                if j != 0:
                    loose_teeth_indices.append(new_distance.index(distance/avg_distance)+j)
        else:
            new_distance.append(distance)

    
    return loose_teeth_indices, new_distance
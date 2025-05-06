import craft_utils
import imgproc
from PIL import Image, ExifTags

import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt


"""
CRAFT Inference Utilities for Text Detection

This module includes functions to perform:
- Text region detection using the CRAFT model
- Orientation correction based on EXIF metadata
- Box merging for overlapping text regions
- Visualization utilities for detected boxes
"""


def run_craft_inference(image_path, net, refine_net, canvas_size = 1280, text_threshold = 0.7, low_text = 0.4, link_threshold = 0.7, mag_ratio = 1.5):

    """
    Runs the CRAFT text detector (with optional refiner) on the input image.

    Args:
        image_path (str): Path to the image file.
        net (CRAFT): Loaded CRAFT model.
        refine_net (RefineNet or None): Optional refinement model.
        canvas_size (int): Maximum image size for inference.
        text_threshold (float): Confidence threshold for text regions.
        low_text (float): Lower bound text confidence.
        link_threshold (float): Threshold for linking characters.
        mag_ratio (float): Magnification ratio for resizing.
    
    Returns:
        Tuple[np.ndarray, List[List[List[int]]]]: Processed image and detected polygon boxes.
    """
    
    image = imgproc.loadImage(image_path)
    image = correct_orientation_cv(image_path,image)
    img_resized, target_ratio, _ = imgproc.resize_aspect_ratio(image, canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=mag_ratio)
    ratio_h = ratio_w = 1 / target_ratio
    x = imgproc.normalizeMeanVariance(img_resized)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).float()
    x = x.cuda() if torch.cuda.is_available() else x

    with torch.no_grad():
        y, feature = net(x)
        score_text = y[0, :, :, 0].cpu().numpy()
        score_link = y[0, :, :, 1].cpu().numpy()

    if refine_net is not None:
        with torch.no_grad():
            y_refiner = refine_net(y, feature)
        score_link = y_refiner[0, :, :, 0].cpu().numpy()

    boxes, polys = craft_utils.getDetBoxes(score_text, score_link, text_threshold, link_threshold, low_text, False)
    boxes = craft_utils.adjustResultCoordinates(boxes, ratio_w, ratio_h)
    polys = craft_utils.adjustResultCoordinates(polys, ratio_w, ratio_h)
    for k in range(len(polys)):
        if polys[k] is None:
            polys[k] = boxes[k]

    return image, polys



def draw_boxes(image, boxes):

    """
    Draws green polygons around detected text regions on the image.

    Args:
        image (np.ndarray): Input image.
        boxes (List[List[List[int]]]): List of polygon boxes.

    Returns:
        np.ndarray: Image with drawn polygons.
    """
        
    img_copy = image.copy()
    for box in boxes:
        pts = np.array(box, np.int32).reshape((-1, 1, 2))
        cv2.polylines(img_copy, [pts], True, (0, 255, 0), thickness=2)
    return img_copy

def correct_orientation_cv(img_path, img_np):

    """
    Corrects the orientation of an image using EXIF metadata (if available).

    Args:
        img_path (str): Path to the image file.
        img_np (np.ndarray): Image in OpenCV format.

    Returns:
        np.ndarray: Orientation-corrected image.
    """

    try:
        img_pil = Image.open(img_path)
        exif = img_pil._getexif()
        orientation_key = next(k for k, v in ExifTags.TAGS.items() if v == 'Orientation')
        if exif is not None and orientation_key in exif:
            orientation = exif[orientation_key]

            if orientation == 3:
                img_np = cv2.rotate(img_np, cv2.ROTATE_180)
            elif orientation == 6:
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_CLOCKWISE)
            elif orientation == 8:
                img_np = cv2.rotate(img_np, cv2.ROTATE_90_COUNTERCLOCKWISE)
    except Exception as e:
        pass  

    return img_np



def merge_boxes(boxes):

    """
    Merges overlapping or closely positioned text boxes into a single bounding box.

    Args:
        boxes (List[List[List[int]]]): List of polygon boxes.

    Returns:
        List[List[List[int]]]: List of merged bounding boxes.
    """
        
    merged = []
    used = [False] * len(boxes)

    for i, box in enumerate(boxes):
        if used[i]:
            continue
        curr = box
        for j in range(i + 1, len(boxes)):
            if not used[j] and do_overlap(curr, boxes[j]):
                curr = merge(curr, boxes[j])
                used[j] = True
        merged.append(curr)

    return merged


def get_rect(box):

    """
    Returns the rectangular bounding coordinates for a polygon box.

    Args:
        box (List[List[int]]): Polygon box with 4 points.

    Returns:
        Tuple[int, int, int, int]: (xmin, ymin, xmax, ymax) coordinates.
    """

    x_coords = [pt[0] for pt in box]
    y_coords = [pt[1] for pt in box]
    return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

def do_overlap(b1, b2):

    """
    Checks whether two bounding boxes overlap.

    Args:
        b1 (List[List[int]]): First bounding box.
        b2 (List[List[int]]): Second bounding box.

    Returns:
        bool: True if the boxes overlap, False otherwise.
    """
        
    x1_min, y1_min, x1_max, y1_max = get_rect(b1)
    x2_min, y2_min, x2_max, y2_max = get_rect(b2)

    horizontally = x1_min <= x2_max and x2_min <= x1_max
    vertically = y1_min <= y2_max and y2_min <= y1_max

    return horizontally and vertically

def merge(b1, b2):

    """
    Merges two bounding boxes into one that encompasses both.

    Args:
        b1 (List[List[int]]): First bounding box.
        b2 (List[List[int]]): Second bounding box.

    Returns:
        List[List[int]]: Merged bounding box as 4-point polygon.
    """
        
    x1_min, y1_min, x1_max, y1_max = get_rect(b1)
    x2_min, y2_min, x2_max, y2_max = get_rect(b2)
    x_min = min(x1_min, x2_min)
    y_min = min(y1_min, y2_min)
    x_max = max(x1_max, x2_max)
    y_max = max(y1_max, y2_max)
    return [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]]
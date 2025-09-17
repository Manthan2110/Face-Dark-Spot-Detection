import cv2
import numpy as np
from typing import Tuple, List

# Basic edge detection using Canny
def apply_edge_detection(image: np.ndarray, method: str = 'canny'):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    return edges

# Extract surrounding skin region below the eye
def get_surrounding_skin(image, eye_coords, padding_ratio = 0.6):

    if eye_coords is None or len(eye_coords) == 0:
        return np.array([])
    
    # Get bounding box of eye region
    x, y, w, h = cv2.boundingRect(eye_coords.reshape(-1,2))
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)

    # Define skin region below the eye box
    # It avoids eyebrow shadow and upper eyelid.
    x1 = max(0, x - pad_x)
    y1 = min(image.shape[0]-1, y + h)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + int(pad_y*2))

    # Ensure valid region
    if y1 >= y2 or x1 >= x2:
        return np.array([])
    
    return image[y1:y2, x1:x2]

# Extract square eye region around keypoint
def get_eye_region_from_point(point, image_shape, size = 60, image: np.ndarray = None):
    if point is None:
        return np.array([])
    
    x, y = int(point[0]), int(point[1])     
    half = size // 2

    x1 = max(0, x - half)
    y1 = max(0, y - half)
    x2 = min(image_shape[1], x + half)
    y2 = min(image_shape[0], y + half)

    # If image is provided, return the cropped region, otherwise return coordinates
    if image is not None:
        return image[y1:y2, x1:x2]
    return np.array([[x1,y1],[x2,y1],[x2,y2],[x1,y2]])

# Calculate text size for given font and scale
def calculate_text_size(text, font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 1.0, thickness = 2):
    return cv2.getTextSize(text, font, font_scale, thickness)

def find_non_overlapping_position(text, base_pos, occupied_regions, img_shape, font_scale = 1.0, min_font_scale = 0.4):

    # Try to find a position for the text that does not overlap with occupied regions
    current_scale = font_scale
    while current_scale >= min_font_scale:
        (text_w, text_h), baseline = calculate_text_size(text, font_scale=current_scale)
        x, y = base_pos
        
        # Try different positions (original, above, below)
        positions = [
            (x - text_w//2, y),  # centered
            (x - text_w//2, y - text_h - 5),  # above
            (x - text_w//2, y + text_h + 5)   # below
        ]
        
        for px, py in positions:
            # Ensure within image bounds
            if px < 0: px = 0
            if px + text_w > img_shape[1]: px = img_shape[1] - text_w
            if py - text_h < 0: py = text_h
            if py > img_shape[0]: py = img_shape[0]
            
            # Check for overlaps
            text_box = (px, py - text_h, px + text_w, py)

            if not any(regions_overlap(text_box, region) for region in occupied_regions):
                return (px, py), current_scale
        
        # If no position works, try smaller font
        current_scale -= 0.1
        
    # If we get here, return best effort with minimum scale
    return base_pos, min_font_scale

# Check if two rectangles overlap
def regions_overlap(r1, r2):
    return not (r1[2] < r2[0] or r1[0] > r2[2] or r1[1] > r2[3] or r1[3] < r2[1])

# Draw text with background rectangle
def draw_text_with_background(img, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale = 1.0,
                        thickness = 2, text_color=(0,255,0), bg_color=(0,0,0), pad = 5):
    
    # text size
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = int(org[0]), int(org[1])
    rect_x1 = max(x - pad, 0)
    rect_y1 = max(y - text_h - baseline - pad, 0)
    rect_x2 = min(x + text_w + pad, img.shape[1])
    rect_y2 = min(y + pad, img.shape[0])

    # draw filled background rectangle
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), bg_color, -1)
    
    # border for better separation
    cv2.rectangle(img, (rect_x1, rect_y1), (rect_x2, rect_y2), (200,200,200), 1)
    
    # put text over the rectangle
    text_org = (x, y - baseline)
    cv2.putText(img, text, text_org, font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return (rect_x1, rect_y1, rect_x2, rect_y2)  # return the occupied region

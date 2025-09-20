import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from mtcnn import MTCNN
from utils import (
    apply_edge_detection,
    get_surrounding_skin,
    get_eye_region_from_point,
    draw_text_with_background,
    find_non_overlapping_position
)

# MTCNN wrapper for face and landmark detection
class MTCNNDetector:
    def __init__(self):
        self.detector = MTCNN()

    def detect_faces_and_landmarks(self, image: np.ndarray):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)        #Convert BGR to RGB because MTCNN expects RGB

        return self.detector.detect_faces(rgb) or []        #Return empty list if no faces detected

class DarkCircleDetector:
    def __init__(self):
        self.mtcnn = MTCNNDetector()

    #Logic to extract eye region from keypoint
    def extract_eye_region_from_keypoint(self, image: np.ndarray, point: Tuple[int,int], size: int = 80):
        return get_eye_region_from_point(point, image.shape, size=size, image=image)      
    
    # LAB color space Analysis
    # Analyze color differences between eye region and surrounding skin(pigmentation detection)
    def color_analysis_method(self, eye_region: np.ndarray, surrounding_skin: np.ndarray):
        if eye_region.size == 0 or surrounding_skin.size == 0:  
            return 0.0, {}
        
        # Convert to LAB color space
        eye_lab = cv2.cvtColor(eye_region, cv2.COLOR_BGR2LAB)
        skin_lab = cv2.cvtColor(surrounding_skin, cv2.COLOR_BGR2LAB)

        eye_mean = np.mean(eye_lab.reshape(-1,3),axis=0)
        skin_mean = np.mean(skin_lab.reshape(-1,3),axis=0)

        darkness_score = float(skin_mean[0]-eye_mean[0])        # Positive if eye region is darker 
        delta_e = float(np.linalg.norm(eye_mean-skin_mean))     # Euclidean distance in LAB space(Gold Standard for color difference)

        return darkness_score, {"delta_e":delta_e,"darkness_score":darkness_score}

    # Histogram-based thresholding(shadow-based detection)
    def histogram_thresholding_method(self, eye_region):
        if eye_region.size == 0:
            return 0.0, np.zeros((1,1),dtype=np.uint8)
        
        # Convert to grayscale and apply adaptive thresholding
        gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray,(5,5),0)                # Reduce noise
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, 
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)       # Inverted binary
        dark_pixels = int(np.sum(adaptive_thresh==255))         # Count dark pixels

        # Return ratio of dark pixels to total pixels in eye region
        return float(dark_pixels/adaptive_thresh.size), adaptive_thresh

    # Combine scores from different methods to get final severity score
    def calculate_severity_score(self, color_score, hist_score, cnn_score=0.0):
        weights={'color':0.4,'histogram':0.4,'cnn':0.2}
        c=min(max(color_score/50.0,0),1)                    # Normalize color score
        h=min(max(hist_score,0),1)                          # Histogram score   

        # Weighted combination
        final = weights['color']*c+weights['histogram']*h+weights['cnn']*cnn_score

        # Severity categorization
        if final<0.1: sev="Minimal"
        elif final<0.2: sev="Mild"
        elif final<0.4: sev="Moderate"
        elif final<0.6: sev="Severe"    
        else: sev="Very Severe"

        return float(final), sev

    def annotate_face(self, image, box, left_eye, right_eye, left_sev, right_sev, face_num):
        # Draw bounding box around face
        x,y,w,h = [int(v) for v in box]
        cv2.rectangle(image,(x,y),(x+w,y+h),(255,200,0),2)
        
        # Keep track of occupied regions to prevent overlap
        occupied_regions = []
        
        # Add face number at the top of bounding box(e.g., "Face #1")
        face_text = f"Face #{face_num}"
        face_pos = (x + w//2, y - 5)        # centered text
        pos, scale = find_non_overlapping_position(face_text, face_pos, occupied_regions, image.shape, 
                                                 font_scale=0.7, min_font_scale=0.4)
        region = draw_text_with_background(image, face_text, pos,
                                       font_scale=scale, thickness=1,
                                       text_color=(255,200,0), bg_color=(40,40,40))
        occupied_regions.append(region)

        # Draw circles at eye keypoints and add severity labels
        if left_eye and right_eye:
            lx,ly = map(int,left_eye)
            rx,ry = map(int,right_eye)
            cv2.circle(image,(lx,ly),4,(0,255,0),-1)  
            cv2.circle(image,(rx,ry),4,(0,255,0),-1)

            # Left eye severity
            left_text = f"Left: {left_sev[1]} ({left_sev[0]:.2f})"
            left_pos = (lx, ly-10)
            pos, scale = find_non_overlapping_position(left_text, left_pos, occupied_regions, image.shape,
                                                   font_scale=0.6, min_font_scale=0.4)
            region = draw_text_with_background(image, left_text, pos,
                                          font_scale=scale, thickness=1)
            occupied_regions.append(region)

            # Right eye severity
            right_text = f"Right: {right_sev[1]} ({right_sev[0]:.2f})"
            right_pos = (rx, ry-10)
            pos, scale = find_non_overlapping_position(right_text, right_pos, occupied_regions, image.shape,
                                                   font_scale=0.6, min_font_scale=0.4)
            region = draw_text_with_background(image, right_text, pos,
                                          font_scale=scale, thickness=1)
            occupied_regions.append(region)

# ➕ ADDED: Draw rectangles below eyes only
    def draw_dark_spot_rectangles(self, image: np.ndarray, eye_point: Tuple[int, int],
                                  eye_region: np.ndarray, mask: np.ndarray, offset: int = 5):  # added
        if mask is None or mask.size == 0:
            return
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 20:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            ex, ey = int(eye_point[0] - mask.shape[1] // 2), int(eye_point[1] - mask.shape[0] // 2)
            # ➕ ADDED: Only draw if the spot is below the eye centerline
            if (ey + y) > (eye_point[1] + offset):  # added
                cv2.rectangle(image, (ex + x, ey + y), (ex + x + w, ey + y + h), (0, 0, 255), 2)  # added

    # Main processing function for an input image
    def process_image(self, image, edge_method):
        try:
            #Face and landmark detection
            detections=self.mtcnn.detect_faces_and_landmarks(image)
            if not detections: return {'success':False,'error':'No faces detected'}

            #For each detected face, analyze dark circles
            annotated=image.copy(); faces=[]
            for i, det in enumerate(detections, 1):
                box=det['box']; k=det['keypoints']
                le,re=k.get('left_eye'),k.get('right_eye')

                # Extract eye regions and surrounding skin
                le_reg=self.extract_eye_region_from_keypoint(image,le) if le else np.array([])
                re_reg=self.extract_eye_region_from_keypoint(image,re) if re else np.array([])
                le_skin=get_surrounding_skin(image, get_eye_region_from_point(le, image.shape)) if le else np.array([])
                re_skin=get_surrounding_skin(image, get_eye_region_from_point(re, image.shape)) if re else np.array([])

                # Left eye analysis
                le_col,_=self.color_analysis_method(le_reg,le_skin)
                le_hist,le_mask=self.histogram_thresholding_method(le_reg)
                le_sev=self.calculate_severity_score(le_col,le_hist)
                
                # Right eye analysis
                re_col,_=self.color_analysis_method(re_reg,re_skin)
                re_hist,re_mask=self.histogram_thresholding_method(re_reg)
                re_sev=self.calculate_severity_score(re_col,re_hist)

                # Annotate results on image
                self.annotate_face(annotated,box,le,re,le_sev,re_sev,i)

                #added: Draw rectangles for left and right dark spots below eyes
                if le is not None: 
                    self.draw_dark_spot_rectangles(annotated, le, le_reg, le_mask)
                if re is not None: 
                    self.draw_dark_spot_rectangles(annotated, re, re_reg, re_mask)

                # Collect face data
                faces.append({'box':box,'confidence':det.get('confidence'),
                              'left_eye_point':le,'right_eye_point':re,
                              'left_severity':le_sev,'right_severity':re_sev})

            # Edge detection 
            edges=apply_edge_detection(image, edge_method)

            return {'success':True, 'annotated_image':annotated, 'edge_image':edges, 'faces':faces}
        
        except Exception as e:
            return {'success':False,'error':str(e)}

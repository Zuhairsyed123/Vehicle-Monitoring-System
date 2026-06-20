import cv2
import numpy as np

class AttributeDetector:
    def __init__(self):
        """
        Initializes the AttributeDetector.
        """
        pass

    def detect_color(self, frame, bbox):
        """
        Detects the dominant color of the vehicle within the bounding box.
        Uses K-Means clustering on the center portion of the bounding box
        to ignore background noise.
        """
        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        
        # Clip bbox to frame dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            return "Unknown"
            
        # Crop the vehicle bounding box
        crop = frame[y1:y2, x1:x2]
        
        # Focus on the center region of the vehicle (60% width and 60% height)
        # to avoid wheels, road, and other background elements.
        ch, cw, _ = crop.shape
        start_x = int(cw * 0.2)
        end_x = int(cw * 0.8)
        start_y = int(ch * 0.2)
        end_y = int(ch * 0.8)
        
        if (end_x - start_x) <= 0 or (end_y - start_y) <= 0:
            # Fallback to whole crop if too small
            center_crop = crop
        else:
            center_crop = crop[start_y:end_y, start_x:end_x]
            
        # Resize to small size (e.g. 30x30) to speed up K-Means significantly
        small_crop = cv2.resize(center_crop, (30, 30))
        
        # Reshape to a 2D array of pixels for cv2.kmeans
        pixels = small_crop.reshape((-1, 3))
        pixels = np.float32(pixels)
        
        # Define criteria and run K-Means (K=3 clusters)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_RANDOM_CENTERS
        compactness, labels, centers = cv2.kmeans(pixels, 3, None, criteria, 10, flags)
        
        # Find the dominant cluster label
        counts = np.bincount(labels.flatten())
        dominant_label = np.argmax(counts)
        
        # Dominant color in BGR format
        dominant_bgr = centers[dominant_label]
        
        # Convert dominant BGR to HSV
        bgr_pixel = np.uint8([[dominant_bgr]])
        hsv_pixel = cv2.cvtColor(bgr_pixel, cv2.COLOR_BGR2HSV)[0][0]
        
        h_val, s_val, v_val = hsv_pixel
        
        # Map HSV values to human-readable color name
        # Hue range: 0-180, Saturation: 0-255, Value: 0-255
        color_name = self._classify_hsv(h_val, s_val, v_val)
        return color_name

    def _classify_hsv(self, h, s, v):
        """
        Classifies color using HSV boundaries.
        """
        # Under daylight/shadows, white/silver cars often have bluish reflection.
        # Check for bright surfaces with low-to-moderate saturation.
        if v > 185 and s < 65:
            return "White"
        if v > 220 and s < 90:
            return "White"
            
        # Low saturation -> Achromatic (White, Gray, Black)
        if s < 35:
            if v > 180:
                return "White"
            elif v < 55:
                return "Black"
            else:
                return "Gray"
        
        # Low value (very dark) -> Black / Dark Gray
        if v < 45:
            return "Black"
            
        # High saturation and value -> Chromatic colors
        # Hue boundaries
        if h < 10 or h >= 170:
            return "Red"
        elif 10 <= h < 22:
            # Distinguish brown from orange based on brightness/saturation
            if v < 120 or s < 100:
                return "Brown"
            return "Orange"
        elif 22 <= h < 38:
            if v < 90 or s < 80:
                return "Brown"
            return "Yellow"
        elif 38 <= h < 85:
            return "Green"
        elif 85 <= h < 130:
            return "Blue"
        elif 130 <= h < 160:
            return "Purple"
        elif 160 <= h < 170:
            return "Pink"
            
        return "Gray"

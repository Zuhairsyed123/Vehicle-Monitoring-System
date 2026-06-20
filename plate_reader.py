import cv2
import re
import easyocr
import logging

# Disable easyocr default verbose logging
logging.getLogger('easyocr').setLevel(logging.ERROR)

class PlateReader:
    def __init__(self, use_gpu=True):
        """
        Initializes the PlateReader with EasyOCR.
        It automatically detects and uses CUDA if available and requested.
        """
        # Load EasyOCR reader for English language
        self.reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # Cache to store plate read results: {track_id: (plate_text, confidence)}
        self.plate_cache = {}
        # Track the last frame index OCR was run for a vehicle: {track_id: last_frame_run}
        self.last_ocr_frame = {}
        # Interval (in frames) to wait before re-running OCR on the same vehicle
        self.ocr_interval = 15

    def clean_plate_text(self, text):
        """
        Removes spaces, special characters, and forces uppercase.
        """
        # Keep only alphanumeric characters
        cleaned = re.sub(r'[^A-Za-z0-9]', '', text)
        return cleaned.upper()

    def is_valid_plate(self, text):
        """
        Heuristic to check if text resembles a license plate.
        Typically, license plates are between 4 and 12 characters, and
        contain a mix of letters and digits.
        """
        if not (4 <= len(text) <= 12):
            return False
            
        # Optional: check if there's at least one digit and one letter
        has_digit = any(c.isdigit() for c in text)
        has_letter = any(c.isalpha() for c in text)
        
        # In some cases plates could be all digits or letters, but having both is very common
        # Let's be slightly flexible: either has both, or is alphanumeric and length is >= 5
        return (has_digit and has_letter) or len(text) >= 5

    def read_plate(self, frame, bbox, track_id, frame_idx):
        """
        Crops the lower half of the vehicle, runs EasyOCR, and returns the license plate text.
        Applies a frame interval and confidence caching mechanism for performance.
        """
        # 1. Return cached plate if we already have a strong reading
        if track_id in self.plate_cache:
            cached_text, cached_conf = self.plate_cache[track_id]
            if cached_conf > 0.65 and self.is_valid_plate(cached_text):
                return cached_text
                
        # 2. Rate-limit OCR calls: only run once every `ocr_interval` frames per vehicle ID
        last_run = self.last_ocr_frame.get(track_id, -self.ocr_interval)
        if frame_idx - last_run < self.ocr_interval:
            # Return current cache (even if weak) or empty string
            if track_id in self.plate_cache:
                return self.plate_cache[track_id][0]
            return ""
            
        # Update last run frame index
        self.last_ocr_frame[track_id] = frame_idx
        
        # 3. Crop lower 50% of the vehicle bounding box where plates are located
        x1, y1, x2, y2 = bbox
        h, w, _ = frame.shape
        
        # Clip bbox to frame dimensions
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        box_h = y2 - y1
        if box_h <= 0 or (x2 - x1) <= 0:
            return ""
            
        # Crop lower half
        lower_y1 = y1 + int(box_h * 0.45)
        crop = frame[lower_y1:y2, x1:x2]
        
        if crop.size == 0:
            return ""
            
        try:
            # 4. Perform OCR text detection and recognition
            results = self.reader.readtext(crop)
            
            best_plate = ""
            best_conf = 0.0
            
            for (ocr_box, text, confidence) in results:
                cleaned = self.clean_plate_text(text)
                
                # Check if this text meets our plate heuristics
                if self.is_valid_plate(cleaned):
                    # We prefer plates that contain both letters and digits,
                    # or select the one with the highest confidence
                    if confidence > best_conf:
                        best_plate = cleaned
                        best_conf = confidence
            
            # 5. Update cache if we found a valid plate
            if best_plate:
                # If we had a previous cache, see if the new one is better
                if track_id in self.plate_cache:
                    prev_text, prev_conf = self.plate_cache[track_id]
                    # Keep the one with higher confidence
                    if best_conf > prev_conf:
                        self.plate_cache[track_id] = (best_plate, best_conf)
                else:
                    self.plate_cache[track_id] = (best_plate, best_conf)
                    
            # Return current cached plate text or empty string
            if track_id in self.plate_cache:
                return self.plate_cache[track_id][0]
                
        except Exception as e:
            # Fail silently to avoid crashing the pipeline
            pass
            
        return ""

    def cleanup(self, active_ids):
        """Cleans up cache for vehicles no longer tracked."""
        inactive_ids = [tid for tid in list(self.plate_cache.keys()) if tid not in active_ids]
        for tid in inactive_ids:
            if tid in self.plate_cache:
                del self.plate_cache[tid]
            if tid in self.last_ocr_frame:
                del self.last_ocr_frame[tid]

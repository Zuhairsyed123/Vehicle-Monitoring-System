import cv2
from ultralytics import YOLO

class VehicleDetector:
    def __init__(self, model_name="yolov8n.pt"):
        """
        Initializes the YOLOv8 vehicle detector.
        Model will automatically download to the models/ directory if not present.
        """
        # Load YOLOv8 model
        self.model = YOLO(model_name)
        
        # COCO vehicle classes: 2: car, 3: motorcycle, 5: bus, 7: truck
        self.vehicle_classes = [2, 3, 5, 7]
        self.class_names = {
            2: "Car",
            3: "Motorcycle",
            5: "Bus",
            7: "Truck"
        }

    def detect(self, frame, conf_threshold=0.3):
        """
        Performs object detection on a single frame.
        Returns:
            List of dicts containing:
                'box': [x1, y1, x2, y2] (integers)
                'confidence': float
                'class_id': int
                'class_name': str
        """
        results = self.model.predict(source=frame, conf=conf_threshold, classes=self.vehicle_classes, verbose=False)
        detections = []
        
        if not results:
            return detections
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            detections.append({
                "box": [x1, y1, x2, y2],
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, "Vehicle")
            })
            
        return detections

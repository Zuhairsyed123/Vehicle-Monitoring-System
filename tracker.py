class VehicleTracker:
    def __init__(self, detector):
        """
        Initializes the tracker using the shared detector's YOLOv8 model instance.
        """
        self.detector = detector
        self.model = detector.model
        self.vehicle_classes = detector.vehicle_classes
        self.class_names = detector.class_names

    def track(self, frame, conf_threshold=0.3):
        """
        Runs ByteTrack on the frame to track vehicles.
        Returns:
            List of dicts containing:
                'box': [x1, y1, x2, y2] (integers)
                'track_id': int (unique tracking ID)
                'confidence': float
                'class_id': int
                'class_name': str
        """
        # YOLOv8 track method runs detection and tracking.
        # persist=True ensures tracking IDs are maintained across frames.
        results = self.model.track(
            source=frame, 
            persist=True, 
            tracker="bytetrack.yaml", 
            classes=self.vehicle_classes, 
            conf=conf_threshold, 
            verbose=False
        )
        
        tracked_objects = []
        if not results:
            return tracked_objects
            
        result = results[0]
        boxes = result.boxes
        
        for box in boxes:
            # Check if tracker ID is assigned
            if box.id is None:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            track_id = int(box.id[0].cpu().numpy())
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            
            tracked_objects.append({
                "box": [x1, y1, x2, y2],
                "track_id": track_id,
                "confidence": conf,
                "class_id": cls_id,
                "class_name": self.class_names.get(cls_id, "Vehicle")
            })
            
        return tracked_objects

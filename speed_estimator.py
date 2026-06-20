import math
import cv2
import numpy as np
from collections import defaultdict

class SpeedEstimator:
    def __init__(self, fps=30.0, speed_limit_kmh=60.0, src_points=None, road_width=10.0, road_length=40.0):
        """
        Initializes the SpeedEstimator with homography-based perspective calibration.
        Args:
            fps (float): Frames Per Second of the video.
            speed_limit_kmh (float): Speed limit in km/h.
            src_points (list): 4 points [[x1,y1], [x2,y2], [x3,y3], [x4,y4]] forming the trapezoid calibration zone on the road.
                               Order: Bottom-Left, Top-Left, Top-Right, Bottom-Right.
            road_width (float): Physical width of the road segment in meters.
            road_length (float): Physical length of the road segment in meters.
        """
        self.fps = fps
        self.speed_limit = speed_limit_kmh
        
        # Default source points for standard traffic camera perspective on a 1280x720 canvas
        if src_points is None:
            self.src_points = [[130, 720], [450, 200], [830, 200], [1150, 720]]
        else:
            self.src_points = src_points
            
        self.road_width = road_width
        self.road_length = road_length
        
        # Dictionary to store BEV position history: {track_id: [(frame_idx, (u, v))]}
        self.history = defaultdict(list)
        # Dictionary to store calculated speeds: {track_id: speed_kmh}
        self.current_speeds = {}
        # Max history length to keep for each vehicle
        self.max_history = 30
        # Window size of frames to calculate displacement
        self.window_size = 10
        
        # Compute Homography Matrix
        self._compute_homography()

    def _compute_homography(self):
        """Computes the perspective transformation matrix M mapping source coordinates to birds-eye view meters."""
        try:
            # Source points in the frame (Float32)
            src = np.float32(self.src_points)
            
            # Destination points in meters space (Flat birds-eye view rectangle)
            # Order maps to: Bottom-Left, Top-Left, Top-Right, Bottom-Right
            dst = np.float32([
                [0, self.road_length],               # Bottom-Left
                [0, 0],                              # Top-Left
                [self.road_width, 0],                # Top-Right
                [self.road_width, self.road_length]  # Bottom-Right
            ])
            
            self.M = cv2.getPerspectiveTransform(src, dst)
            self.homography_valid = True
        except Exception as e:
            print(f"Error computing homography matrix: {e}")
            self.homography_valid = False

    def update_parameters(self, fps=None, speed_limit_kmh=None, src_points=None, road_width=None, road_length=None):
        """Updates configurations and recalculates homography matrix dynamically."""
        params_changed = False
        
        if fps is not None:
            self.fps = fps
        if speed_limit_kmh is not None:
            self.speed_limit = speed_limit_kmh
            
        if src_points is not None and src_points != self.src_points:
            self.src_points = src_points
            params_changed = True
        if road_width is not None and road_width != self.road_width:
            self.road_width = road_width
            params_changed = True
        if road_length is not None and road_length != self.road_length:
            self.road_length = road_length
            params_changed = True
            
        if params_changed:
            self._compute_homography()

    def project_point(self, x, y):
        """Projects a camera frame point (x, y) into birds-eye view meter space (u, v) using homography."""
        if not self.homography_valid:
            # Return scaled dummy values if homography fails
            return x * 0.05, y * 0.05
            
        point = np.array([[[x, y]]], dtype=np.float32)
        bev_point = cv2.perspectiveTransform(point, self.M)[0][0]
        u, v = bev_point
        return float(u), float(v)

    def estimate_speed(self, track_id, bbox, frame_idx):
        """
        Updates position history in meter space and estimates speed for a vehicle using perspective homography.
        Args:
            track_id (int): Tracking ID of the vehicle.
            bbox (list): Bounding box [x1, y1, x2, y2].
            frame_idx (int): Current frame index.
        Returns:
            float: Estimated speed in km/h.
        """
        x1, y1, x2, y2 = bbox
        
        # Use the bottom center of the vehicle bounding box (point on the road surface)
        cx = (x1 + x2) / 2
        cy = y2
        
        # Project frame point to BEV meter space
        u, v = self.project_point(cx, cy)
        
        # Append current position
        self.history[track_id].append((frame_idx, (u, v)))
        
        # Clean up history if it grows too large
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id].pop(0)
            
        # If history has fewer than 2 frames, speed cannot be calculated yet
        history_len = len(self.history[track_id])
        if history_len < 2:
            return self.current_speeds.get(track_id, 0.0)
            
        # Compare current BEV position with position 'window' frames ago
        window = min(history_len, self.window_size)
        start_frame, start_pos = self.history[track_id][-window]
        end_frame, end_pos = self.history[track_id][-1]
        
        # Compute distance in physical meters
        du = end_pos[0] - start_pos[0]
        dv = end_pos[1] - start_pos[1]
        distance_meters = math.sqrt(du**2 + dv**2)
        
        # Time elapsed in seconds
        frames_elapsed = end_frame - start_frame
        if frames_elapsed <= 0:
            return self.current_speeds.get(track_id, 0.0)
            
        time_seconds = frames_elapsed / self.fps
        
        # Calculate speed: speed (m/s) = meters / seconds
        speed_mps = distance_meters / time_seconds
        speed_kmh = speed_mps * 3.6
        
        # Noise filter: If displacement is negligible, vehicle is stationary
        if (distance_meters / frames_elapsed) < 0.015:
            speed_kmh = 0.0
            
        # Apply smoothing: 75% old speed, 25% new speed to filter tracking noise
        if track_id in self.current_speeds:
            prev_speed = self.current_speeds[track_id]
            if prev_speed > 0:
                speed_kmh = 0.75 * prev_speed + 0.25 * speed_kmh
            
        # Round speed
        self.current_speeds[track_id] = round(speed_kmh, 1)
        return self.current_speeds[track_id]


    def is_violating(self, speed_kmh):
        """Checks if speed violates the speed limit."""
        return speed_kmh > self.speed_limit

    def cleanup(self, active_ids):
        """Cleans up history for vehicles no longer tracked."""
        inactive_ids = [tid for tid in list(self.history.keys()) if tid not in active_ids]
        for tid in inactive_ids:
            if tid in self.history:
                del self.history[tid]
            if tid in self.current_speeds:
                del self.current_speeds[tid]

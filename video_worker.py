import os
import cv2
import csv
import time
import numpy as np
from datetime import datetime
from database import TrafficDatabase
from detector import VehicleDetector
from tracker import VehicleTracker
from speed_estimator import SpeedEstimator
from attribute_detector import AttributeDetector
from plate_reader import PlateReader

def draw_overlay_text(image, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, scale=0.45, color=(255, 255, 255), thickness=1, bg_color=(0, 0, 0)):
    """Draws overlay text with a dark background for legibility."""
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = position
    cv2.rectangle(image, (x - 4, y - text_h - 4), (x + text_w + 4, y + baseline + 4), bg_color, -1)
    cv2.putText(image, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return text_h + baseline + 8

def process_uploaded_video(video_id, input_path, output_path, speed_limit=60.0, src_points=None, road_width=10.0, road_length=40.0):
    """
    Worker function to process an uploaded video in the background.
    Logs progress to database, updates statistics, exports processed video, CSV logs, and text summaries.
    Computes speed using Homography/perspective transformation matrix.
    """
    db = TrafficDatabase()
    db.update_video_progress(video_id, 1.0, status="processing")
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Unable to open uploaded video at {input_path}")
        db.update_video_progress(video_id, 0.0, status="failed")
        return
        
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if total_frames <= 0:
        total_frames = 100 # safety fallback
        
    # Set up OpenCV VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    try:
        # Load models
        detector = VehicleDetector()
        tracker = VehicleTracker(detector)
        
        # Instantiate Homography Speed Estimator
        speed_estimator = SpeedEstimator(
            fps=fps, 
            speed_limit_kmh=speed_limit, 
            src_points=src_points, 
            road_width=road_width, 
            road_length=road_length
        )
        
        attribute_detector = AttributeDetector()
        plate_reader = PlateReader(use_gpu=True)
        
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_idx += 1
            
            # Draw Perspective Calibration Polygon on Video
            if src_points:
                pts = np.array(src_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 165, 255), thickness=2) # Draw Orange Calibration trapezoid
                cv2.putText(frame, "CALIBRATION ZONE", (src_points[1][0], src_points[1][1] - 8), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
            
            # Run tracking pipeline
            tracked_objects = tracker.track(frame, conf_threshold=0.3)
            active_ids = []
            
            for obj in tracked_objects:
                x1, y1, x2, y2 = obj['box']
                track_id = obj['track_id']
                class_name = obj['class_name']
                confidence = obj['confidence']
                active_ids.append(track_id)
                
                # Estimate speed using homography
                speed = speed_estimator.estimate_speed(track_id, [x1, y1, x2, y2], frame_idx)
                violating = speed_estimator.is_violating(speed)
                
                # Detect color
                color = attribute_detector.detect_color(frame, [x1, y1, x2, y2])
                
                # Read license plate
                plate = plate_reader.read_plate(frame, [x1, y1, x2, y2], track_id, frame_idx)
                
                # Log vehicle details in DB
                db.log_vehicle(track_id, class_name, color, speed, plate, violating, video_id=video_id)
                
                # Capture speed violations
                if violating:
                    crop_x1 = max(0, x1 - 10)
                    crop_y1 = max(0, y1 - 10)
                    crop_x2 = min(width, x2 + 10)
                    crop_y2 = min(height, y2 + 10)
                    vehicle_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2].copy()
                    
                    if vehicle_crop.size > 0:
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        img_name = f"violation_vid_{video_id}_{track_id}_{timestamp_str}.jpg"
                        img_path = os.path.join("violations", img_name)
                        cv2.imwrite(img_path, vehicle_crop)
                        db.log_violation(track_id, class_name, color, speed, plate, img_path, video_id=video_id)
                
                # Draw boxes and overlays on output frame
                box_color = (0, 0, 239) if violating else (0, 220, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                overlay_x = x1
                overlay_y = y1 - 10
                
                text_id = f"ID: {track_id} | {class_name} ({int(confidence*100)}%)"
                text_attr = f"Color: {color} | Plate: {plate if plate else 'Scanning...'}"
                text_speed = f"Speed: {speed:.1f} km/h"
                
                bg_color = (0, 0, 150) if violating else (0, 100, 0)
                
                # Stack HUD labels
                overlay_y -= draw_overlay_text(frame, text_id, (overlay_x, overlay_y), bg_color=bg_color)
                overlay_y -= draw_overlay_text(frame, text_attr, (overlay_x, overlay_y), bg_color=bg_color)
                
                if violating:
                    overlay_y -= draw_overlay_text(frame, "!!! OVERSPEEDING !!!", (overlay_x, overlay_y), color=(10, 10, 255), thickness=2, bg_color=(0, 0, 200))
                
                draw_overlay_text(frame, text_speed, (x1, y1 - 10 - 45), color=(255, 255, 255), thickness=1, bg_color=bg_color)
                
            # Clean tracking state
            speed_estimator.cleanup(active_ids)
            plate_reader.cleanup(active_ids)
            
            # Write frame to output video file
            out.write(frame)
            
            # Update progress percentage every 10 frames
            if frame_idx % 10 == 0:
                progress = round((frame_idx / total_frames) * 100, 1)
                progress = min(progress, 99.0)  # Keep 100.0 for final save
                db.update_video_progress(video_id, progress)
                
        # Finalize processing
        cap.release()
        out.release()
        
        # Calculate summary statistics for the video run
        conn = db.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT type, max_speed, violation_status FROM vehicles WHERE video_id = ?", (video_id,))
        rows = cursor.fetchall()
        
        total_vehicles = len(rows)
        
        avg_speed = 0.0
        max_speed = 0.0
        min_speed = 999.0
        overspeeding_count = 0
        
        cars_count = 0
        bikes_count = 0
        buses_count = 0
        trucks_count = 0
        
        if total_vehicles > 0:
            speeds = []
            for r in rows:
                v_type = r["type"]
                speed = r["max_speed"]
                viol = r["violation_status"]
                
                speeds.append(speed)
                if speed > max_speed:
                    max_speed = speed
                if speed > 0.0 and speed < min_speed:
                    min_speed = speed
                if viol == 1:
                    overspeeding_count += 1
                    
                if v_type == "Car":
                    cars_count += 1
                elif v_type == "Motorcycle":
                    bikes_count += 1
                elif v_type == "Bus":
                    buses_count += 1
                elif v_type == "Truck":
                    trucks_count += 1
            
            avg_speed = round(sum(speeds) / total_vehicles, 1)
            max_speed = round(max_speed, 1)
            min_speed = round(min_speed, 1) if min_speed != 999.0 else 0.0
        else:
            min_speed = 0.0
            
        stats = {
            "total_vehicles": total_vehicles,
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "min_speed": min_speed,
            "overspeeding_count": overspeeding_count,
            "cars_count": cars_count,
            "bikes_count": bikes_count,
            "buses_count": buses_count,
            "trucks_count": trucks_count
        }
        
        # Generate CSV report file
        csv_filename = f"report_video_{video_id}.csv"
        csv_path = os.path.join("logs", "reports", csv_filename)
        
        with open(csv_path, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Vehicle_ID", "Vehicle_Type", "Color", "Max_Speed_KMH", "Plate_Number", "Timestamp", "Violation_Status"])
            
            cursor.execute("""
                SELECT vehicle_id, type, color, max_speed, plate_number, timestamp, violation_status
                FROM vehicles WHERE video_id = ? ORDER BY vehicle_id ASC
            """, (video_id,))
            v_rows = cursor.fetchall()
            for vr in v_rows:
                writer.writerow([
                    vr["vehicle_id"],
                    vr["type"],
                    vr["color"],
                    round(vr["max_speed"], 1),
                    vr["plate_number"] if vr["plate_number"] else "N/A",
                    vr["timestamp"],
                    "Overspeed" if vr["violation_status"] == 1 else "Safe"
                ])
                
        # --- GENERATE SUMMARY TEXT REPORT ---
        # Fetch color distribution dynamically
        cursor.execute("""
            SELECT color, COUNT(*) as count FROM vehicles 
            WHERE video_id = ? GROUP BY color ORDER BY count DESC
        """, (video_id,))
        color_rows = cursor.fetchall()
        color_dist_str = ""
        for cr in color_rows:
            color_pct = (cr["count"] / total_vehicles * 100) if total_vehicles > 0 else 0.0
            color_dist_str += f"- {cr['color']}: {cr['count']} ({color_pct:.1f}%)\n"
            
        # Fetch basic video metadata
        cursor.execute("SELECT filename, upload_time FROM videos WHERE id = ?", (video_id,))
        vid_row = cursor.fetchone()
        raw_filename = vid_row["filename"] if vid_row else "unknown_video.mp4"
        upload_time = vid_row["upload_time"] if vid_row else "N/A"
        clean_filename = '_'.join(raw_filename.split('_')[2:]) if '_' in raw_filename else raw_filename

        
        summary_filename = f"summary_video_{video_id}.txt"
        summary_path = os.path.join("logs", "reports", summary_filename)
        
        violation_pct = (overspeeding_count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        car_pct = (cars_count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        bike_pct = (bikes_count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        bus_pct = (buses_count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        truck_pct = (trucks_count / total_vehicles * 100) if total_vehicles > 0 else 0.0
        
        summary_content = f"""=============================================
TRAFFIC SURVEILLANCE ANALYTICS REPORT
=============================================
Video Name: {clean_filename}
Upload Date: {upload_time}
Surveillance Run ID: {video_id}
Status: Completed Analysis

=============================================
SUMMARY METRICS
=============================================
Total Vehicles Detected: {total_vehicles}
Average Vehicle Speed  : {avg_speed} km/h
Fastest Speed Recorded : {max_speed} km/h
Slowest Speed Recorded : {min_speed} km/h
Configured Speed Limit : {speed_limit} km/h
Total Speed Violations : {overspeeding_count} ({violation_pct:.1f}% of traffic flow)

=============================================
VEHICLE CLASS DISTRIBUTION
=============================================
Cars       : {cars_count} ({car_pct:.1f}%)
Motorcycles: {bikes_count} ({bike_pct:.1f}%)
Buses      : {buses_count} ({bus_pct:.1f}%)
Trucks     : {trucks_count} ({truck_pct:.1f}%)

=============================================
VEHICLE COLOR DISTRIBUTION
=============================================
{color_dist_str if color_dist_str else "- No color metrics recorded."}
=============================================
Generated by Smart Vehicle Speed and Attribute Detection System.
"""
        with open(summary_path, mode='w') as sf:
            sf.write(summary_content)
            
        conn.close()
        
        # Complete video entry in database
        db.complete_video(video_id, output_path, stats)
        print(f"Asynchronous processing of video {video_id} completed successfully.")
        
    except Exception as e:
        print(f"Error processing video {video_id} in background worker: {e}")
        db.update_video_progress(video_id, 0.0, status="failed")
        if cap.isOpened():
            cap.release()
        try:
            out.write(frame)
        except:
            pass

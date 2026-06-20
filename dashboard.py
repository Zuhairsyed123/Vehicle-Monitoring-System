import os
import json
import threading
from datetime import datetime
from flask import Flask, render_template, Response, jsonify, request, send_from_directory, make_response
from database import TrafficDatabase
from video_worker import process_uploaded_video

app = Flask(__name__, template_folder='templates')

# Folder Configurations
UPLOAD_FOLDER = 'videos/uploads'
OUTPUT_FOLDER = 'videos/outputs'
REPORT_FOLDER = 'logs/reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

class PipelineState:
    def __init__(self):
        # Default perspective coordinates (1280x720)
        self.src_points = [[130, 720], [450, 200], [830, 200], [1150, 720]]
        self.road_width = 10.0  # physical width of road in meters
        self.road_length = 40.0  # physical length of road in meters
        self.speed_limit = 60.0
        self.db = TrafficDatabase()

state = PipelineState()

@app.route('/')
def index():
    """Renders the main upload & analysis page."""
    return render_template('index.html')

# --- VIDEO UPLOAD & ANALYTICS MODULE ENDPOINTS ---

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handles traffic video uploads and spawns the background processing worker."""
    if 'video' not in request.files:
        return jsonify({"status": "error", "message": "No video file provided"}), 400
        
    file = request.files['video']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected"}), 400
        
    speed_limit = float(request.form.get('speed_limit', state.speed_limit))
    
    # Parse perspective points
    src_points_str = request.form.get('src_points')
    if src_points_str:
        try:
            src_points = json.loads(src_points_str)
        except Exception:
            src_points = state.src_points
    else:
        src_points = state.src_points
        
    road_width = float(request.form.get('road_width', state.road_width))
    road_length = float(request.form.get('road_length', state.road_length))
    
    # Save the uploaded file
    filename = secure_filename_local(file.filename)
    timestamp_str = datetime_now_str()
    unique_filename = f"{timestamp_str}_{filename}"
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)
    
    # Output file configuration
    output_filename = f"processed_{unique_filename}"
    output_filepath = os.path.join(OUTPUT_FOLDER, output_filename)
    
    # Register video in DB
    video_id = state.db.add_video(unique_filename, filepath)
    
    if video_id is None:
        return jsonify({"status": "error", "message": "Failed to create database session"}), 500
        
    # Start background processing thread with Homography params
    t = threading.Thread(
        target=process_uploaded_video,
        args=(video_id, filepath, output_filepath, speed_limit, src_points, road_width, road_length),
        daemon=True
    )
    t.start()
    
    return jsonify({
        "status": "success",
        "video_id": video_id,
        "filename": unique_filename
    })

@app.route('/api/video_status/<int:video_id>')
def get_video_status(video_id):
    """Returns the current status, progress percentage, and unique vehicles detected so far."""
    video = state.db.get_video_by_id(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video session not found"}), 404
    return jsonify(video)

@app.route('/api/video_report/<int:video_id>')
def get_video_report(video_id):
    """Returns statistics, categories, colors, and violations logs for a specific processed video."""
    video = state.db.get_video_by_id(video_id)
    if not video:
        return jsonify({"status": "error", "message": "Video report not found"}), 404
        
    recent_violations = state.db.get_recent_violations(limit=100, video_id=video_id)
    category_counts = state.db.get_vehicle_counts_by_type(video_id=video_id)
    
    # Fetch color counts from DB dynamically
    conn = state.db.get_connection()
    cursor = conn.cursor()
    color_counts = {}
    try:
        cursor.execute("SELECT color, COUNT(*) as count FROM vehicles WHERE video_id = ? GROUP BY color ORDER BY count DESC", (video_id,))
        rows = cursor.fetchall()
        for row in rows:
            color_counts[row["color"]] = row["count"]
    except Exception as e:
        print(f"Error querying color distribution: {e}")
    finally:
        conn.close()
    
    response_data = {
        "video_info": video,
        "category_counts": category_counts,
        "color_counts": color_counts,
        "recent_violations": recent_violations
    }
    return jsonify(response_data)

@app.route('/download/video/<int:video_id>')
def download_processed_video(video_id):
    """Downloads the processed annotated output video file."""
    video = state.db.get_video_by_id(video_id)
    if not video or not video["output_filepath"]:
        return "Processed video not found.", 404
        
    output_dir = os.path.dirname(video["output_filepath"])
    output_file = os.path.basename(video["output_filepath"])
    return send_from_directory(output_dir, output_file, as_attachment=True)

@app.route('/stream/video/<int:video_id>')
def stream_processed_video(video_id):
    """Streams the processed annotated output video file inline in the browser."""
    video = state.db.get_video_by_id(video_id)
    if not video or not video["output_filepath"]:
        return "Processed video not found.", 404
        
    output_dir = os.path.dirname(video["output_filepath"])
    output_file = os.path.basename(video["output_filepath"])
    return send_from_directory(output_dir, output_file)


@app.route('/download/report/<int:video_id>')
def download_csv_report(video_id):
    """Downloads the compiled CSV log file."""
    video = state.db.get_video_by_id(video_id)
    if not video or video["status"] != "completed":
        return "Report is not ready yet.", 400
        
    report_filename = f"report_video_{video_id}.csv"
    return send_from_directory(REPORT_FOLDER, report_filename, as_attachment=True)

@app.route('/download/summary/<int:video_id>')
def download_summary_report(video_id):
    """Downloads the text summary analytics report file."""
    video = state.db.get_video_by_id(video_id)
    if not video or video["status"] != "completed":
        return "Summary report is not ready yet.", 400
        
    summary_filename = f"summary_video_{video_id}.txt"
    return send_from_directory(REPORT_FOLDER, summary_filename, as_attachment=True)

@app.route('/api/history')
def get_upload_history():
    """Returns a list of all uploaded videos and processing statuses."""
    videos = state.db.get_all_videos()
    return jsonify(videos)

# Custom static files server to serve violation screenshots from the root violations/ folder
@app.route('/static/violations/<path:filename>')
def serve_violation_image(filename):
    """Serves evidence images directly from the root violations/ directory."""
    return send_from_directory('violations', filename)

# --- UTILITIES ---

def secure_filename_local(filename):
    """Removes path separators for safety and returns a standard filename."""
    import re
    # Keep alphanumeric, dots, dashes, and underscores
    clean = re.sub(r'[^A-Za-z0-9._-]', '_', filename)
    return clean

def datetime_now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

if __name__ == '__main__':
    state.db.init_db()
    app.run(host='0.0.0.0', port=5000, debug=True)

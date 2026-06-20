import sqlite3
import os
from datetime import datetime

class TrafficDatabase:
    def __init__(self, db_path="logs/traffic_surveillance.db"):
        """
        Initializes the SQLite database.
        Creates tables if they do not exist.
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def get_connection(self):
        """Returns a connection to the SQLite database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Returns results as dict-like objects
        return conn

    def init_db(self):
        """Creates the database schema if tables do not exist."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Table to store uploaded video sessions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                filepath TEXT,
                output_filepath TEXT,
                upload_time TEXT,
                status TEXT, -- 'pending', 'processing', 'completed', 'failed'
                progress REAL DEFAULT 0.0,
                total_vehicles INTEGER DEFAULT 0,
                avg_speed REAL DEFAULT 0.0,
                max_speed REAL DEFAULT 0.0,
                min_speed REAL DEFAULT 0.0,
                overspeeding_count INTEGER DEFAULT 0,
                cars_count INTEGER DEFAULT 0,
                bikes_count INTEGER DEFAULT 0,
                buses_count INTEGER DEFAULT 0,
                trucks_count INTEGER DEFAULT 0
            )
        """)
        
        # Table to store unique vehicle logs (added video_id column)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER,
                video_id INTEGER DEFAULT 0, -- 0 represents live feed
                type TEXT,
                color TEXT,
                max_speed REAL,
                plate_number TEXT,
                timestamp TEXT,
                violation_status INTEGER DEFAULT 0,
                UNIQUE(vehicle_id, video_id)
            )
        """)
        
        # Table to store specific speed violations (added video_id column)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                vehicle_id INTEGER,
                video_id INTEGER DEFAULT 0, -- 0 represents live feed
                type TEXT,
                color TEXT,
                speed REAL,
                plate_number TEXT,
                timestamp TEXT,
                image_path TEXT
            )
        """)
        
        conn.commit()
        conn.close()

    def log_vehicle(self, vehicle_id, vehicle_type, color, speed, plate_number, violation_status, video_id=0):
        """
        Logs or updates a vehicle's record in the database.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            # Check if vehicle exists within the same video stream/session
            cursor.execute("""
                SELECT max_speed, plate_number, violation_status FROM vehicles 
                WHERE vehicle_id = ? AND video_id = ?
            """, (vehicle_id, video_id))
            row = cursor.fetchone()
            
            if row:
                # Update existing record
                existing_max_speed = row["max_speed"]
                new_max_speed = max(existing_max_speed, speed)
                
                existing_plate = row["plate_number"]
                new_plate = plate_number if plate_number and (not existing_plate or len(plate_number) > len(existing_plate)) else existing_plate
                
                new_violation = 1 if (row["violation_status"] == 1 or violation_status) else 0
                
                cursor.execute("""
                    UPDATE vehicles 
                    SET type = ?, color = ?, max_speed = ?, plate_number = ?, violation_status = ? 
                    WHERE vehicle_id = ? AND video_id = ?
                """, (vehicle_type, color, new_max_speed, new_plate, new_violation, vehicle_id, video_id))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO vehicles (vehicle_id, video_id, type, color, max_speed, plate_number, timestamp, violation_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (vehicle_id, video_id, vehicle_type, color, speed, plate_number, timestamp, 1 if violation_status else 0))
                
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error logging vehicle: {e}")
        finally:
            conn.close()

    def log_violation(self, vehicle_id, vehicle_type, color, speed, plate_number, image_path, video_id=0):
        """
        Logs a speed violation and saves evidence record. Stores only image filename.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Store only the base image filename for platform-independent serving
        image_filename = os.path.basename(image_path)
        
        try:
            # Prevent double-logging the same violation for the same vehicle in a short time
            cursor.execute("""
                SELECT timestamp FROM violations 
                WHERE vehicle_id = ? AND video_id = ?
                ORDER BY id DESC LIMIT 1
            """, (vehicle_id, video_id))
            row = cursor.fetchone()
            
            should_log = True
            if row:
                prev_time = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
                curr_time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                if (curr_time - prev_time).total_seconds() < 15:
                    should_log = False
            
            if should_log:
                cursor.execute("""
                    INSERT INTO violations (vehicle_id, video_id, type, color, speed, plate_number, timestamp, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (vehicle_id, video_id, vehicle_type, color, speed, plate_number, timestamp, image_filename))
                conn.commit()
        except sqlite3.Error as e:
            print(f"Database error logging violation: {e}")
        finally:
            conn.close()

    def get_stats(self, video_id=0):
        """
        Returns stats metrics for a specific video run (or 0 for live).
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {
            "total_vehicles": 0,
            "avg_speed": 0.0,
            "violations_count": 0
        }
        
        try:
            cursor.execute("SELECT COUNT(*) FROM vehicles WHERE video_id = ?", (video_id,))
            stats["total_vehicles"] = cursor.fetchone()[0]
            
            cursor.execute("SELECT AVG(max_speed) FROM vehicles WHERE video_id = ?", (video_id,))
            avg_speed = cursor.fetchone()[0]
            stats["avg_speed"] = round(avg_speed, 1) if avg_speed else 0.0
            
            cursor.execute("SELECT COUNT(*) FROM violations WHERE video_id = ?", (video_id,))
            stats["violations_count"] = cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"Database error getting stats: {e}")
        finally:
            conn.close()
            
        return stats

    def get_recent_violations(self, limit=10, video_id=0):
        """
        Returns recent violations for a specific video run (or 0 for live).
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        violations = []
        
        try:
            cursor.execute("""
                SELECT id, vehicle_id, type, color, speed, plate_number, timestamp, image_path 
                FROM violations 
                WHERE video_id = ?
                ORDER BY id DESC LIMIT ?
            """, (video_id, limit))
            rows = cursor.fetchall()
            for row in rows:
                violations.append(dict(row))
        except sqlite3.Error as e:
            print(f"Database error getting recent violations: {e}")
        finally:
            conn.close()
            
        return violations

    def get_vehicle_counts_by_type(self, video_id=0):
        """
        Returns vehicle count binned by class for a specific video run.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}
        
        try:
            cursor.execute("SELECT type, COUNT(*) as count FROM vehicles WHERE video_id = ? GROUP BY type", (video_id,))
            rows = cursor.fetchall()
            for row in rows:
                v_type = row["type"]
                if v_type in counts:
                    counts[v_type] = row["count"]
        except sqlite3.Error as e:
            print(f"Database error getting counts: {e}")
        finally:
            conn.close()
            
        return counts

    # --- VIDEO SESSION MANAGEMENT METHODS ---

    def add_video(self, filename, filepath):
        """Inserts a new video log entry, setting status to pending."""
        conn = self.get_connection()
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        video_id = None
        try:
            cursor.execute("""
                INSERT INTO videos (filename, filepath, upload_time, status, progress)
                VALUES (?, ?, ?, 'pending', 0.0)
            """, (filename, filepath, timestamp))
            conn.commit()
            video_id = cursor.lastrowid
        except sqlite3.Error as e:
            print(f"Database error adding video: {e}")
        finally:
            conn.close()
        return video_id

    def update_video_progress(self, video_id, progress, status="processing"):
        """Updates the status and percentage progress of a video."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE videos 
                SET progress = ?, status = ?
                WHERE id = ?
            """, (progress, status, video_id))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error updating progress: {e}")
        finally:
            conn.close()

    def complete_video(self, video_id, output_filepath, stats):
        """Completes a video log, updating finalized stats fields."""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                UPDATE videos 
                SET status = 'completed',
                    progress = 100.0,
                    output_filepath = ?,
                    total_vehicles = ?,
                    avg_speed = ?,
                    max_speed = ?,
                    min_speed = ?,
                    overspeeding_count = ?,
                    cars_count = ?,
                    bikes_count = ?,
                    buses_count = ?,
                    trucks_count = ?
                WHERE id = ?
            """, (
                output_filepath,
                stats.get("total_vehicles", 0),
                stats.get("avg_speed", 0.0),
                stats.get("max_speed", 0.0),
                stats.get("min_speed", 0.0),
                stats.get("overspeeding_count", 0),
                stats.get("cars_count", 0),
                stats.get("bikes_count", 0),
                stats.get("buses_count", 0),
                stats.get("trucks_count", 0),
                video_id
            ))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error completing video: {e}")
        finally:
            conn.close()

    def get_video_by_id(self, video_id):
        """Returns details of a specific video session."""
        conn = self.get_connection()
        cursor = conn.cursor()
        video_data = None
        try:
            cursor.execute("SELECT * FROM videos WHERE id = ?", (video_id,))
            row = cursor.fetchone()
            if row:
                video_data = dict(row)
                # Fetch unique vehicles detected so far for this video
                cursor.execute("SELECT COUNT(*) FROM vehicles WHERE video_id = ?", (video_id,))
                video_data["vehicles_detected"] = cursor.fetchone()[0]
        except sqlite3.Error as e:
            print(f"Database error getting video: {e}")
        finally:
            conn.close()
        return video_data


    def get_all_videos(self):
        """Returns a list of all processed video uploads."""
        conn = self.get_connection()
        cursor = conn.cursor()
        videos = []
        try:
            cursor.execute("SELECT * FROM videos ORDER BY id DESC")
            rows = cursor.fetchall()
            for row in rows:
                videos.append(dict(row))
        except sqlite3.Error as e:
            print(f"Database error getting all videos: {e}")
        finally:
            conn.close()
        return videos

    def get_video_vehicles(self, video_id):
        """Returns list of tracked vehicles for a specific video id (for CSV exports)."""
        conn = self.get_connection()
        cursor = conn.cursor()
        vehicles = []
        try:
            cursor.execute("""
                SELECT vehicle_id, type, color, max_speed, plate_number, timestamp, violation_status
                FROM vehicles
                WHERE video_id = ?
                ORDER BY vehicle_id ASC
            """, (video_id,))
            rows = cursor.fetchall()
            for row in rows:
                vehicles.append(dict(row))
        except sqlite3.Error as e:
            print(f"Database error getting video vehicles: {e}")
        finally:
            conn.close()
        return vehicles

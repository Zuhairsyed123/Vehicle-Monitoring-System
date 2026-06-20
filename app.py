import os
from dashboard import app, state

if __name__ == '__main__':
    # Ensure necessary folders exist
    os.makedirs('violations', exist_ok=True)
    os.makedirs('logs/reports', exist_ok=True)
    os.makedirs('videos/uploads', exist_ok=True)
    os.makedirs('videos/outputs', exist_ok=True)
    
    # Initialize SQLite Database
    state.db.init_db()
    
    # Start Flask Webserver on Port 5000
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)



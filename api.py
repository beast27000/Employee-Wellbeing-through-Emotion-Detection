from fastapi import FastAPI
import psycopg
from psycopg.rows import dict_row
from datetime import datetime

app = FastAPI()

# Emotion mapping
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Database connection function
def get_db_connection():
    try:
        conn = psycopg.connect(
            dbname="emotion_detection",
            user="postgres",
            password="Calcite*1234",
            host="localhost",
            row_factory=dict_row
        )
        return conn
    except psycopg.Error as e:
        raise Exception(f"Database connection failed: {e}")

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion Detection API"}

# Get all emotions
@app.get("/emotions")
def get_all_emotions():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM emotions ORDER BY date_stamp, time_stamp")
    results = cur.fetchall()
    # Map emotion numbers to names
    for row in results:
        row['emotion_name'] = emotions[row['emotion']]
    cur.close()
    conn.close()
    return results

# Get emotions by user ID
@app.get("/emotions/user/{user_id}")
def get_emotions_by_user(user_id: str):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM emotions WHERE id = %s ORDER BY date_stamp, time_stamp", (user_id,))
    results = cur.fetchall()
    for row in results:
        row['emotion_name'] = emotions[row['emotion']]
    cur.close()
    conn.close()
    return results

# Get emotions by date
@app.get("/emotions/date/{date}")
def get_emotions_by_date(date: str):
    try:
        datetime.strptime(date, "%Y-%m-%d")  # Validate date format
    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD"}
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM emotions WHERE date_stamp = %s ORDER BY time_stamp", (date,))
    results = cur.fetchall()
    for row in results:
        row['emotion_name'] = emotions[row['emotion']]
    cur.close()
    conn.close()
    return results

# Run the app (for testing)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
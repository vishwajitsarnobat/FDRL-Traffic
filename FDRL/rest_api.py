# rest_api.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from collections import deque
from fastapi.middleware.cors import CORSMiddleware

# --- Data Validation Model ---
class LogEntry(BaseModel):
    junction_id: str
    timestamp: float
    queue_lengths: List[int]
    waiting_times: List[float]
    current_action: int

# --- In-memory Buffer ---
# Use a deque to automatically discard old entries
MAX_LOG_SIZE = 1000
log_buffer = deque(maxlen=MAX_LOG_SIZE)

# --- FastAPI App ---
app = FastAPI(title="FDRL Traffic Control API")

# --- CORS Middleware (allows browser frontend to access API) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/logs")
def receive_log(entry: LogEntry):
    """
    Endpoint to receive a single log entry from the inference script.
    """
    log_buffer.append(entry.dict())
    return {"status": "success", "message": f"Log for {entry.junction_id} received."}

@app.get("/get_latest_logs")
def get_latest_logs():
    """
    Endpoint for a frontend to fetch the latest buffered log data.
    """
    return list(log_buffer)

# To run this server:
# uvicorn rest_api:app --host 127.0.0.1 --port 8000 --reload
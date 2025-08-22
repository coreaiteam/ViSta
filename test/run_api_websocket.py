# run_api_websocket.py
from fastapi import WebSocket, WebSocketDisconnect, Request
from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.service.service import ClusteringService
from app.service.data_ingestion import DataSourceType
from app.database.base import SessionLocal
from app.service.models import UserLocation
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Clustering Service API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
clustering_service = None
websocket_manager = None
api_state_manager = None


class ConnectionManager:
    def __init__(self):
        # Map user_id to WebSocket
        self.active_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int):
        await websocket.accept()
        # store connection by user_id
        self.active_connections[user_id] = websocket

    def disconnect(self, user_id: int):
        if user_id in self.active_connections:
            del self.active_connections[user_id]  # remove connection

    async def broadcast(self, message: dict):
        for connection in self.active_connections.values():
            await connection.send_json(message)

    async def send_personal_message(self, user_id: int, message: dict):
        websocket = self.active_connections.get(user_id)
        if websocket:
            await websocket.send_json(message)


class APIStateManager:
    def __init__(self):
        self.current_state = {}

    def update_state(self, message):
        # Store the latest state for API endpoints
        if message.message_type == "group_formed":
            self.current_state[message.data["group_id"]] = message.data


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize managers
    websocket_manager = ConnectionManager()
    api_state_manager = APIStateManager()

    # Create service
    clustering_service = ClusteringService(clustering_interval=5)

    # Add database ingestor
    clustering_service.add_data_ingestor(
        DataSourceType.DATABASE,
        {"session_factory": SessionLocal}
    )

    # Add WebSocket output handler
    clustering_service.add_output_handler({
        "type": "websocket",
        "connection_manager": websocket_manager
    })

    # Add API output handler
    clustering_service.add_output_handler({
        "type": "api",
        "api_state_manager": api_state_manager
    })

    # Start the service
    clustering_service.start()
    logger.info("Clustering service started with API and WebSocket support")

    # Store references in app.state (instead of globals)
    app.state.clustering_service = clustering_service
    app.state.websocket_manager = websocket_manager
    app.state.api_state_manager = api_state_manager

    yield

    # Shutdown
    if clustering_service:
        clustering_service.stop()
        logger.info("Clustering service stopped")


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    websocket_manager = app.state.websocket_manager
    clustering_service = app.state.clustering_service

    group = clustering_service.get_user_group(user_id)

    await websocket_manager.connect(websocket, user_id)

    await websocket_manager.send_personal_message(user_id, group)

    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_manager.disconnect(user_id)


@app.post("/users/")
async def add_user_location(
    user_id: int,
    origin_lat: float,
    origin_lng: float,
    destination_lat: float,
    destination_lng: float,
    request: Request
):
    clustering_service = app.state.clustering_service

    from datetime import datetime

    user_location = UserLocation(
        user_id=user_id,
        origin_lat=origin_lat,
        origin_lng=origin_lng,
        destination_lat=destination_lat,
        destination_lng=destination_lng,
        stored_at=datetime.now()
    )

    clustering_service.add_user_location(user_location)

    return {"status": "success", "user_id": user_id}


@app.get("/groups/")
async def get_all_groups(request: Request):
    clustering_service = request.app.state.clustering_service
    if clustering_service:
        return clustering_service.get_all_active_groups()
    return []


@app.get("/status/")
async def get_service_status(request: Request):
    clustering_service = request.app.state.clustering_service
    if clustering_service:
        return clustering_service.get_service_status()
    return {"status": "service not initialized"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

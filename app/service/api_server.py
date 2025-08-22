from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import List, Dict
from .models import UserLocation
from .service import ClusteringService
from contextlib import asynccontextmanager


app = FastAPI()

# Global service instance
_clustering_service: ClusteringService = None
_websocket_manager = None
_api_state_manager = None


def get_clustering_service() -> ClusteringService:
    global _clustering_service
    if _clustering_service is None:
        _clustering_service = ClusteringService()

        # Add database ingestor
        from app.database import SessionLocal
        _clustering_service.add_data_ingestor(
            "database",
            {"session_factory": SessionLocal}
        )

        # Add output handlers
        _clustering_service.add_output_handler({
            "type": "websocket",
            "connection_manager": _websocket_manager
        })

        _clustering_service.add_output_handler({
            "type": "api",
            "api_state_manager": _api_state_manager
        })

    return _clustering_service


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.user_connections: Dict[int, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: int = None):
        await websocket.accept()
        self.active_connections.append(websocket)
        if user_id is not None:
            self.user_connections[user_id] = websocket

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        # Remove from user_connections if present
        user_id = None
        for uid, conn in self.user_connections.items():
            if conn == websocket:
                user_id = uid
                break
        if user_id:
            del self.user_connections[user_id]

    async def send_personal_message(self, user_id: int, message: Dict):
        if user_id in self.user_connections:
            await self.user_connections[user_id].send_json(message)

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            await connection.send_json(message)


class APIStateManager:
    def __init__(self):
        self.current_state = {}

    def update_state(self, message):
        message_type = message.message_type
        data = message.data

        if message_type == "group_formed":
            group_id = data["group_id"]
            self.current_state[group_id] = data["group_data"]
        elif message_type == "group_updated":
            group_id = data["group_id"]
            self.current_state[group_id] = data["group_data"]
        elif message_type == "group_disbanded":
            group_id = data["group_id"]
            if group_id in self.current_state:
                del self.current_state[group_id]


# Initialize managers
_websocket_manager = ConnectionManager()
_api_state_manager = APIStateManager()


@app.post("/users/")
async def add_user_location(user_location: UserLocation):
    service = get_clustering_service()
    service.add_user_location(user_location)
    return {"status": "success", "user_id": user_location.user_id}


@app.get("/users/{user_id}/group")
async def get_user_group(user_id: int):
    service = get_clustering_service()
    group = service.get_user_group(user_id)
    return group if group else {"status": "user not in a group"}


@app.get("/users/{user_id}/companions")
async def get_user_companions(user_id: int):
    service = get_clustering_service()
    companions = service.get_user_companions(user_id)
    return companions if companions else {"status": "user not in a group"}


@app.get("/groups/")
async def get_all_groups():
    service = get_clustering_service()
    return service.get_all_active_groups()


@app.get("/status/")
async def get_service_status():
    service = get_clustering_service()
    return service.get_service_status()


@app.delete("/users/{user_id}")
async def remove_user(user_id: int):
    service = get_clustering_service()
    success = service.remove_user(user_id)
    return {"status": "success" if success else "user not found"}


@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: int):
    await _websocket_manager.connect(websocket, user_id)
    try:
        while True:
            # Keep connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        _websocket_manager.disconnect(websocket)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    service = get_clustering_service()
    service.start()

    yield  # Application runs while inside here

    # Shutdown
    service.stop()

app = FastAPI(lifespan=lifespan)

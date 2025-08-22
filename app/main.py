from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database.base import Base, engine, SessionLocal
from app.routes import home, ride, schedule, history
from app.auth.router import router as auth_router
from app.config import settings
from contextlib import asynccontextmanager
from app.service.service import get_clustering_service
from app.service.data_ingestion import DataSourceType
from app.service.connection_manager import get_websocket_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        clustering_service = get_clustering_service(clustering_interval=5)
        websocket_manager = get_websocket_manager()

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

        clustering_service.start()

        yield
    finally:
        # Shutdown
        if clustering_service:
            clustering_service.stop()

app = FastAPI(title="ViSta",  lifespan=lifespan, debug=settings.DEBUG)

# Setup database
Base.metadata.create_all(bind=engine)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
app.include_router(auth_router)
app.include_router(home.router)
app.include_router(ride.router)
app.include_router(schedule.router)
app.include_router(history.router)


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "RideShare is running"}

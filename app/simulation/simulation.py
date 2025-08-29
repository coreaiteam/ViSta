from ..service.service import get_clustering_service
from .generator import OSMTestDataPipeline
from ..service.data_ingestion import DataSourceType
from ..service.models import UserLocation
from ..database.base import SessionLocal
import osmnx as ox
import time
from datetime import datetime, timezone
from ..service.connection_manager import get_websocket_manager
from .visualization.main import app

PLACE = "Savojbolagh County, Alborz Province, Iran"


def _callback(message):
    """Simple output handler"""

    print('===== message : ', end=' ')
    print(message)
    print('----------------------------------------------------')



def get_service():
    clustering_service = get_clustering_service(clustering_interval=5)
    clustering_service.start()
    return clustering_service


def generate_data():
    graph = ox.graph_from_place(PLACE, network_type='walk')

    pipeline = OSMTestDataPipeline(
        graph,
        storage_path="savojbolagh.json",
        num_main_points=3,
        neighbors_k=20,
        sample_size=20,
        max_walk_dist=200,
        seed=42
    )

    pipeline.prepare(force_recompute=True)
    dataset = pipeline.get_dataset()

    return dataset


def loc2userlocation(user_id, loc):
    return UserLocation(
        user_id=user_id,
        origin_lat=loc[0],
        origin_lng=loc[1],
        destination_lat=loc[2],
        destination_lng=loc[3],
        stored_at=datetime.now(timezone.utc),
    )


def main():
    service = get_service()
    data = generate_data()

    # Add database ingestor
    # service.add_data_ingestor(
    #     DataSourceType.DATABASE,
    #     {"sessi5on_factory": SessionLocal}
    # )

    # Add WebSocket output handler
    websocket_manager = get_websocket_manager()
    service.add_output_handler({
        "type": "websocket",
        "connection_manager": websocket_manager
    })
    

    # service.add_output_handler({
    #     "type": "callback",
    #     "callback": _callback
    # })
    try:
        # Keep the service running
        app.run()
        while True:
            for user_id, loc in enumerate(data):
                service.add_user_location(loc2userlocation(user_id, loc))
                time.sleep(2)
            time.sleep(10)
    except KeyboardInterrupt:
        service.stop()


if __name__ == "__main__":
    main()

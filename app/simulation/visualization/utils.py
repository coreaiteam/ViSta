import random
from datetime import datetime, timezone
from typing import Tuple, List, Dict

from networkx import MultiDiGraph
import osmnx as ox

from app.service.models import UserLocation
from app.simulation.generator import OSMTestDataPipeline

from ...config import PLACE


def generate_random_users(
    n: int,
    origin_center: Tuple[float, float],
    destination_center: Tuple[float, float],
    spread: float = 0.01,
) -> List[Dict]:
    """
    Generate user users with origins and destinations clustered in different areas of Tehran.

    Args:
        n: Number of users to generate
        origin_center: Center point for origin cluster (lat, lng)
        destination_center: Center point for destination cluster (lat, lng)
        spread: Spread of the cluster in degrees (approximately 1.1 km per 0.01 degrees)

    Returns:
        List of UserLocation objects
    """
    users = []
    for i in range(n):
        # Generate origin points clustered around origin_center
        origin_lat = origin_center[0] + random.uniform(-spread, spread)
        origin_lng = origin_center[1] + random.uniform(-spread, spread)

        # Generate destination points clustered around destination_center
        dest_lat = destination_center[0] + random.uniform(-spread, spread)
        dest_lng = destination_center[1] + random.uniform(-spread, spread)

        users.append(
            UserLocation(
                user_id=i + 1,
                origin_lat=origin_lat,
                origin_lng=origin_lng,
                destination_lat=dest_lat,
                destination_lng=dest_lng,
                stored_at=datetime.now(timezone.utc),
            ).to_dict()
        )
    return users



# def generate_data():
#     graph = ox.graph_from_place(PLACE, network_type='walk')

#     pipeline = OSMTestDataPipeline(
#         graph,
#         storage_path="savojbolagh.json",
#         num_main_points=3,
#         neighbors_k=20,
#         sample_size=20,
#         max_walk_dist=200,
#         seed=42
#     )

#     pipeline.prepare(force_recompute=True)
#     dataset = pipeline.get_dataset()

#     return dataset

# Modified to return graph as well for calculating metrics
def generate_data(graph: MultiDiGraph = None, return_graph: bool = False):
    if not graph:
        graph = ox.graph_from_place(PLACE, network_type='walk')

    pipeline = OSMTestDataPipeline(
        graph,
        storage_path="savojbolagh.json",
        num_main_points=250,
        neighbors_k=20,
        sample_size=5000,
        max_walk_dist=500,
        seed=42
    )

    pipeline.prepare(force_recompute=True)
    dataset = pipeline.get_dataset()
    
    if return_graph:
        return dataset, graph  # Return both dataset and graph
    else:
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
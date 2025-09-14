# # 1. With existing graph (FAST - no download)
# from .visualization.utils import generate_data
# from .metrics import evaluate_user_clustering

# dataset = generate_data()
# metrics = evaluate_user_clustering(
#     user_locations=users,
#     clusters=clusters,
#     graph=dataset.graph  # Use existing graph
# )

# # 2. Without graph but with place name (SLOW - will download)
# metrics = evaluate_user_clustering(
#     user_locations=users,
#     clusters=clusters,
#     place_name="Savojbolagh County, Alborz Province, Iran"
# )

# # 3. With default place name (SLOW - will download)
# metrics = evaluate_user_clustering(
#     user_locations=users,
#     clusters=clusters
#     # Uses default place_name from function definition
# )
# test_metrics.py - Fixed version
import numpy as np
from typing import List
from datetime import datetime
import logging
import networkx as nx
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your modules
from .visualization.utils import generate_data
from .metrics import evaluate_user_clustering

# Mock UserLocation class
class UserLocation:
    def __init__(self, user_id, origin_lat, origin_lng, destination_lat, destination_lng, stored_at, status="PENDING"):
        self.user_id = user_id
        self.origin_lat = origin_lat
        self.origin_lng = origin_lng
        self.destination_lat = destination_lat
        self.destination_lng = destination_lng
        self.stored_at = stored_at
        self.status = status

def test_with_existing_graph():
    """Test with graph from existing pipeline"""
    print("=== TEST WITH EXISTING GRAPH ===")
    
    # Generate data using your pipeline - get both dataset and graph
    dataset, graph = generate_data(return_graph=True)
    
    print(f"Graph has {len(graph.nodes())} nodes")
    print(f"Dataset has {len(dataset.data)} entries")
    
    # Create test data from dataset
    user_locations = []
    for i, (origin_lat, origin_lng, dest_lat, dest_lng) in enumerate(dataset.data[:10]):  # Use first 10 entries
        user = UserLocation(
            user_id=i + 1,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            destination_lat=dest_lat,
            destination_lng=dest_lng,
            stored_at=datetime.now()
        )
        user_locations.append(user)
        print(f"User {i+1}: Origin ({origin_lat:.6f}, {origin_lng:.6f}), Destination ({dest_lat:.6f}, {dest_lng:.6f})")
    
    # Create clusters
    clusters = [user_locations[:5], user_locations[5:]]
    print(f"Created {len(clusters)} clusters with sizes: {[len(cluster) for cluster in clusters]}")
    
    # Evaluate USING EXISTING GRAPH (no download)
    try:
        metrics = evaluate_user_clustering(
            user_locations=user_locations,
            clusters=clusters,
            graph=graph  # Pass existing graph
        )
        
        print(f"\n=== EVALUATION RESULTS ===")
        print(f"Using existing graph: {len(graph.nodes())} nodes")
        print(f"Computation time: {metrics.computation_time:.2f} seconds")
        print(f"Combined SSE: {metrics.combined_sse:.2f}")
        print(f"Origin SSE: {metrics.origin_metrics.sse:.2f}")
        print(f"Destination SSE: {metrics.destination_metrics.sse:.2f}")
        print(f"Cluster sizes: {metrics.cluster_sizes}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def test_without_graph():
    """Test without existing graph (will download)"""
    print("=== TEST WITHOUT EXISTING GRAPH ===")
    
    # Create random test data
    user_locations = []
    for i in range(10):
        user = UserLocation(
            user_id=i + 1,
            origin_lat=35.8 + random.uniform(-0.01, 0.01),
            origin_lng=50.9 + random.uniform(-0.01, 0.01),
            destination_lat=35.8 + random.uniform(-0.01, 0.01),
            destination_lng=50.9 + random.uniform(-0.01, 0.01),
            stored_at=datetime.now()
        )
        user_locations.append(user)
    
    # Create clusters
    clusters = [user_locations[:5], user_locations[5:]]
    
    # Evaluate WITHOUT existing graph (will download)
    try:
        metrics = evaluate_user_clustering(
            user_locations=user_locations,
            clusters=clusters,
            place_name="Savojbolagh County, Alborz Province, Iran"  # Will download this area
        )
        
        print(f"SSE: {metrics.combined_sse:.2f}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

def test_with_place_name_only():
    """Test with just place name"""
    print("=== TEST WITH PLACE NAME ONLY ===")
    
    # Create test data
    user_locations = []
    for i in range(5):
        user = UserLocation(
            user_id=i + 1,
            origin_lat=35.96 + random.uniform(-0.005, 0.005),
            origin_lng=50.73 + random.uniform(-0.005, 0.005),
            destination_lat=35.96 + random.uniform(-0.005, 0.005),
            destination_lng=50.73 + random.uniform(-0.005, 0.005),
            stored_at=datetime.now()
        )
        user_locations.append(user)
    
    # Single cluster
    clusters = [user_locations]
    
    # Evaluate with place name (will download)
    try:
        metrics = evaluate_user_clustering(
            user_locations=user_locations,
            clusters=clusters,
            place_name="Savojbolagh County, Alborz Province, Iran"
        )
        
        print(f"SSE: {metrics.combined_sse:.2f}")
        print("Test completed successfully!")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Run different test scenarios
    test_with_existing_graph()
    print()
    test_without_graph()
    print()
    test_with_place_name_only()
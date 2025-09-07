# test_metrics.py
import numpy as np
from typing import List
from datetime import datetime
from .metrics import evaluate_user_clustering
from ..service.models import UserLocation

# Your test data
all_data = [
    (35.9786555, 50.7312188, 35.9624696, 50.6766717),
    (35.9808466, 50.7465189, 35.983439, 50.7425883),
    # (35.9819054, 50.7471605, 35.9845818, 50.7426107),
    (35.9810731, 50.7476138, 35.9834694, 50.7418251),
    (35.980092, 50.7456665, 35.9844569, 50.7429393),
    # (35.9809762, 50.7462185, 35.9834661, 50.7419666),
    (35.9814105, 50.746828, 35.982993, 50.741938),
    (35.9816071, 50.7463592, 35.9834274, 50.7430993),
    # (35.9820379, 50.7468836, 35.9829962, 50.7417995),
    # (35.9619607, 50.6897019, 35.9588595, 50.6727111),
    # (35.962426, 50.689881, 35.9582907, 50.6721972),
    # (35.9635874, 50.689078, 35.9576577, 50.6721773),
    # (35.9617841, 50.6883783, 35.9578648, 50.6710729),
    # (35.961094, 50.6881121, 35.9575173, 50.6724751),
    # (35.9630772, 50.6874343, 35.9585374, 50.6731009),
    # (35.9625895, 50.6872289, 35.9588, 50.6726574)
]

def create_user_locations() -> List[UserLocation]:
    """Create UserLocation objects from the test data"""
    user_locations = []
    
    for i, (origin_lat, origin_lng, dest_lat, dest_lng) in enumerate(all_data, 1):
        user = UserLocation(
            user_id=i,
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            destination_lat=dest_lat,
            destination_lng=dest_lng,
            stored_at=datetime.now(),
            status="PENDING"  # Assuming UserStatus.PENDING
        )
        user_locations.append(user)
    
    return user_locations

def create_clusters(user_locations: List[UserLocation]) -> List[List[UserLocation]]:
    """Create the specified clustering"""
    # Group users according to your specification
    # clusters = [
    #     [user_locations[0], user_locations[1], user_locations[2]],   # Cluster 0: users 1,2,3
    #     [user_locations[3], user_locations[4], user_locations[5]],   # Cluster 1: users 4,5,6
    #     [user_locations[6], user_locations[7], user_locations[8]],   # Cluster 2: users 7,8,9
    #     [user_locations[9], user_locations[10]],                     # Cluster 3: users 10,11
    #     [user_locations[11], user_locations[12], user_locations[13]], # Cluster 4: users 12,13,14
    #     [user_locations[14], user_locations[15]]                     # Cluster 5: users 15,16
    # ]
    
    clusters = [
        [user_locations[1], user_locations[2], user_locations[3]],   # Cluster 0: users 1,2,3
        [user_locations[0], user_locations[4], user_locations[5]],   # Cluster 1: users 4,5,6
    ]
    return clusters

def test_clustering_evaluation():
    """Test the clustering evaluation with the provided data"""
    print("=== CLUSTERING EVALUATION TEST ===")
    print(f"Testing with {len(all_data)} user locations...")
    
    # Create UserLocation objects
    user_locations = create_user_locations()
    print(f"Created {len(user_locations)} UserLocation objects")
    
    # Create clusters
    clusters = create_clusters(user_locations)
    print(f"Created {len(clusters)} clusters with sizes: {[len(cluster) for cluster in clusters]}")
    
    # Test with different alpha values
    alpha_values = [1.0]
    
    for alpha in alpha_values:
        print(f"\n{'='*50}")
        print(f"EVALUATION WITH ALPHA = {alpha}")
        print(f"{'='*50}")
        
        try:
            # Evaluate clustering
            metrics = evaluate_user_clustering(user_locations, clusters, alpha, "medoid")
            
            # Print results
            print(f"\nComputation time: {metrics.computation_time:.2f} seconds")
            
            print(f"\n=== COMBINED METRICS (alpha={alpha}) ===")
            print(f"Combined SSE: {metrics.combined_sse:.2f}")
            print(f"Combined Intra-cluster distance: {metrics.combined_intra_cluster:.2f}m")
            print(f"Combined Silhouette Score: {metrics.combined_silhouette:.3f}")
            print(f"Dunn Index: {metrics.dun_index:.3f}")
            
            print(f"\n=== ORIGIN METRICS ===")
            print(f"Origin SSE: {metrics.origin_metrics.sse:.2f}")
            print(f"Origin Intra-cluster: {metrics.origin_metrics.intra_cluster:.2f}m")
            print(f"Origin Max Radius: {metrics.origin_metrics.max_radius:.2f}m")
            print(f"Origin Average Radius: {metrics.origin_metrics.average_radius:.2f}m")
            print(f"Origin Silhouette Score: {metrics.origin_metrics.silhouette_score:.3f}")
            
            print(f"\n=== DESTINATION METRICS ===")
            print(f"Destination SSE: {metrics.destination_metrics.sse:.2f}")
            print(f"Destination Intra-cluster: {metrics.destination_metrics.intra_cluster:.2f}m")
            print(f"Destination Max Radius: {metrics.destination_metrics.max_radius:.2f}m")
            print(f"Destination Average Radius: {metrics.destination_metrics.average_radius:.2f}m")
            print(f"Destination Silhouette Score: {metrics.destination_metrics.silhouette_score:.3f}")
            
            print(f"\n=== CLUSTER INFORMATION ===")
            print(f"Cluster sizes: {metrics.cluster_sizes}")
            
            print(f"\n=== INTER-CLUSTER DISTANCES ===")
            for cluster_pair, distance in metrics.inter_cluster_distances.items():
                print(f"Clusters {cluster_pair}: {distance:.2f}m")
            
            print(f"\n=== CENTROID INFORMATION ===")
            for cluster_id in sorted(metrics.origin_centroids.keys()):
                origin_centroid = metrics.origin_centroids[cluster_id]
                dest_centroid = metrics.destination_centroids[cluster_id]
                print(f"Cluster {cluster_id}:")
                print(f"  Origin centroid: ({origin_centroid.origin_lat:.6f}, {origin_centroid.origin_lng:.6f})")
                print(f"  Destination centroid: ({dest_centroid.destination_lat:.6f}, {dest_centroid.destination_lng:.6f})")
        
        except Exception as e:
            print(f"Error during evaluation with alpha={alpha}: {str(e)}")
            import traceback
            traceback.print_exc()

def analyze_cluster_geography(user_locations, clusters):
    """Simple analysis of cluster geography"""
    print(f"\n{'='*50}")
    print("CLUSTER GEOGRAPHICAL ANALYSIS")
    print(f"{'='*50}")
    
    for cluster_idx, cluster in enumerate(clusters):
        print(f"\nCluster {cluster_idx} (size: {len(cluster)}):")
        
        # Origin coordinates
        origin_lats = [user.origin_lat for user in cluster]
        origin_lngs = [user.origin_lng for user in cluster]
        print(f"  Origin range: lat({min(origin_lats):.6f}-{max(origin_lats):.6f}), "
              f"lng({min(origin_lngs):.6f}-{max(origin_lngs):.6f})")
        
        # Destination coordinates
        dest_lats = [user.destination_lat for user in cluster]
        dest_lngs = [user.destination_lng for user in cluster]
        print(f"  Destination range: lat({min(dest_lats):.6f}-{max(dest_lats):.6f}), "
              f"lng({min(dest_lngs):.6f}-{max(dest_lngs):.6f})")

if __name__ == "__main__":
    # Create test data
    user_locations = create_user_locations()
    clusters = create_clusters(user_locations)
    
    # Analyze cluster geography first
    analyze_cluster_geography(user_locations, clusters)
    
    # Run evaluation tests
    test_clustering_evaluation()
# metrics.py
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.spatial import distance
from sklearn.metrics import silhouette_score
import warnings
import osmnx as ox
import networkx as nx
from geopy.distance import great_circle
import time
import logging
from functools import lru_cache

# Import your models
from ..service.models import UserLocation

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PointTypeMetrics:
    """Metrics for a specific point type (origin or destination)"""
    sse: float
    intra_cluster: float
    max_radius: float
    average_radius: float
    silhouette_score: float

@dataclass
class ClusterMetrics:
    """Container for all cluster evaluation metrics with origin-destination integration"""
    # Combined metrics
    combined_sse: float
    combined_intra_cluster: float
    combined_silhouette: float
    dun_index: float
    
    # Point-type specific metrics
    origin_metrics: PointTypeMetrics
    destination_metrics: PointTypeMetrics
    
    # Cluster information
    cluster_sizes: Dict[int, int]
    inter_cluster_distances: Dict[str, float]
    
    # Centroids
    origin_centroids: Dict[int, UserLocation]
    destination_centroids: Dict[int, UserLocation]
    
    # Computation parameters
    computation_time: float
    alpha: float  # Weight for destination in combined metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for easy serialization"""
        return {
            'combined_metrics': {
                'sse': self.combined_sse,
                'intra_cluster': self.combined_intra_cluster,
                'silhouette_score': self.combined_silhouette,
                'dun_index': self.dun_index,
            },
            'origin_metrics': {
                'sse': self.origin_metrics.sse,
                'intra_cluster': self.origin_metrics.intra_cluster,
                'max_radius': self.origin_metrics.max_radius,
                'average_radius': self.origin_metrics.average_radius,
                'silhouette_score': self.origin_metrics.silhouette_score,
            },
            'destination_metrics': {
                'sse': self.destination_metrics.sse,
                'intra_cluster': self.destination_metrics.intra_cluster,
                'max_radius': self.destination_metrics.max_radius,
                'average_radius': self.destination_metrics.average_radius,
                'silhouette_score': self.destination_metrics.silhouette_score,
            },
            'cluster_sizes': self.cluster_sizes,
            'inter_cluster_distances': self.inter_cluster_distances,
            'origin_centroids': {cid: centroid.to_dict() for cid, centroid in self.origin_centroids.items()},
            'destination_centroids': {cid: centroid.to_dict() for cid, centroid in self.destination_centroids.items()},
            'computation_time': self.computation_time,
            'alpha_weight': self.alpha
        }
# class OSMnxDistanceCalculator:
#     """Handles OSMnx graph operations and distance calculations"""
    
#     def __init__(self, network_type='walk'):
#         self.network_type = network_type
#         self.graph = None
    
#     def initialize_graph(self, points: List[Any], buffer_dist=10000):  # Increased buffer
#         """Initialize OSMnx graph for the area covering all points"""
#         try:
#             # Get all coordinates (both origins and destinations)
#             all_coords = []
#             for point in points:
#                 if hasattr(point, 'origin_lat') and hasattr(point, 'origin_lng'):
#                     all_coords.append((point.origin_lat, point.origin_lng))
#                 if hasattr(point, 'destination_lat') and hasattr(point, 'destination_lng'):
#                     all_coords.append((point.destination_lat, point.destination_lng))
            
#             if not all_coords:
#                 raise ValueError("No coordinates found in points")
            
#             lats, lngs = zip(*all_coords)
#             center_lat, center_lng = np.mean(lats), np.mean(lngs)
            
#             logger.info(f"Downloading OSMnx graph for center ({center_lat:.6f}, {center_lng:.6f}) with radius {buffer_dist}m")
            
#             # Download graph - use a larger area to ensure coverage
#             self.graph = ox.graph_from_point(
#                 (center_lat, center_lng), 
#                 dist=buffer_dist, 
#                 network_type=self.network_type,
#                 simplify=True
#             )
            
#             # Project graph
#             self.graph = ox.project_graph(self.graph)
#             logger.info(f"Graph initialized with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
            
#         except Exception as e:
#             logger.error(f"Failed to initialize OSMnx graph: {e}")
#             # Fallback to great circle distance
#             logger.warning("Using great circle distance as fallback")
    
#     def get_walking_distance(self, lat1: float, lng1: float, lat2: float, lng2: float, point_type: str = 'origin') -> float:
#         """Get walking distance between two points with proper debugging"""
#         try:
#             # First check if points are the same
#             if lat1 == lat2 and lng1 == lng2:
#                 return 0.0
            
#             # If graph is not available, use great circle distance
#             if self.graph is None:
#                 distance_meters = great_circle((lat1, lng1), (lat2, lng2)).meters
#                 logger.debug(f"Using great circle distance: {distance_meters:.2f}m")
#                 return distance_meters
            
#             # OSMnx expects (lat, lng) but nearest_nodes expects (lng, lat)
#             # This is the critical fix!
#             point1 = (lat1, lng1)
#             point2 = (lat2, lng2)
            
#             # Get nearest nodes - OSMnx uses (lng, lat) order!
#             try:
#                 node1 = ox.distance.nearest_nodes(self.graph, lng1, lat1)  # (lng, lat)
#                 node2 = ox.distance.nearest_nodes(self.graph, lng2, lat2)  # (lng, lat)
#             except Exception as e:
#                 logger.warning(f"Nearest nodes failed: {e}, using great circle")
#                 return great_circle(point1, point2).meters
            
#             # Check if nodes are the same (points are very close)
#             if node1 == node2:
#                 distance_meters = great_circle(point1, point2).meters
#                 logger.debug(f"Same node, using great circle: {distance_meters:.2f}m")
#                 return distance_meters
            
#             # Calculate shortest path distance
#             try:
#                 distance_meters = nx.shortest_path_length(
#                     self.graph, 
#                     node1, 
#                     node2, 
#                     weight='length'
#                 )
#                 logger.debug(f"OSMnx distance: {distance_meters:.2f}m between {point1} and {point2}")
#                 return distance_meters
                
#             except (nx.NetworkXNoPath, nx.NodeNotFound):
#                 # Fallback to great circle distance if no path found
#                 distance_meters = great_circle(point1, point2).meters
#                 logger.warning(f"No path found between {point1} and {point2}, using great circle: {distance_meters:.0f}m")
#                 return distance_meters
                
#         except Exception as e:
#             logger.error(f"Error calculating walking distance between ({lat1}, {lng1}) and ({lat2}, {lng2}): {e}")
#             # Fallback to great circle distance
#             return great_circle((lat1, lng1), (lat2, lng2)).meters
class OSMnxDistanceCalculator:
    """Handles OSMnx distance calculations with flexible graph initialization"""
    
    def __init__(self, graph: Optional[nx.MultiDiGraph] = None, 
                 network_type: str = 'walk',
                 place_name: Optional[str] = None):
        """
        Initialize OSMnx distance calculator
        
        Args:
            graph: Pre-loaded OSMnx graph (optional)
            network_type: Type of network ('walk', 'drive', etc.)
            place_name: Place name to download graph if not provided
        """
        self.network_type = network_type
        self.place_name = place_name
        self.graph = graph
        
        # Initialize graph if not provided
        if self.graph is None:
            self._initialize_graph_from_place()
        else:
            self._validate_and_prepare_graph()
        
        logger.info(f"OSMnx calculator initialized with {len(self.graph.nodes())} nodes and {len(self.graph.edges())} edges")
    
    def _initialize_graph_from_place(self):
        """Initialize graph by downloading from OSMnx"""
        if not self.place_name:
            raise ValueError("Either graph or place_name must be provided")
        
        try:
            logger.info(f"Downloading OSMnx graph for: {self.place_name}")
            self.graph = ox.graph_from_place(
                self.place_name, 
                network_type=self.network_type,
                simplify=True
            )
            self.graph = ox.project_graph(self.graph)
            logger.info(f"Graph downloaded with {len(self.graph.nodes())} nodes")
            
        except Exception as e:
            logger.error(f"Failed to download graph: {e}")
            raise
    
    def _validate_and_prepare_graph(self):
        """Validate and prepare the provided graph"""
        if self.graph is None:
            raise ValueError("Graph cannot be None")
        
        # Check if graph has CRS information
        if not hasattr(self.graph, 'graph') or 'crs' not in self.graph.graph:
            logger.warning("Graph missing CRS information, attempting to project")
            try:
                self.graph = ox.project_graph(self.graph)
                logger.info("Graph was successfully projected")
            except Exception as e:
                logger.error(f"Failed to project graph: {e}")
                raise
        
        # Alternative method to check if graph is projected
        try:
            # Try to get a sample node to check coordinates
            sample_node = next(iter(self.graph.nodes()))
            sample_data = self.graph.nodes[sample_node]
            x, y = sample_data.get('x', 0), sample_data.get('y', 0)
            
            # If coordinates are in reasonable range, assume projected
            # (Projected coordinates are usually large numbers like meters)
            if abs(x) < 180 and abs(y) < 90:
                logger.info("Graph appears to be in geographic coordinates (lat/lng), projecting...")
                self.graph = ox.project_graph(self.graph)
            else:
                logger.info("Graph appears to be already projected")
                
        except Exception as e:
            logger.warning(f"Could not determine projection status: {e}")
            # Try to project anyway
            try:
                self.graph = ox.project_graph(self.graph)
                logger.info("Graph was projected as fallback")
            except Exception as e:
                logger.error(f"Failed to project graph: {e}")
                raise
    
    def get_walking_distance(self, lat1: float, lng1: float, lat2: float, lng2: float, point_type: str = 'origin') -> float:
        """Get walking distance between two points using the graph"""
        try:
            # First check if points are the same
            if abs(lat1 - lat2) < 1e-6 and abs(lng1 - lng2) < 1e-6:
                return 0.0
            
            # Get nearest nodes - OSMnx uses (lng, lat) order!
            try:
                node1 = ox.distance.nearest_nodes(self.graph, lng1, lat1)
                node2 = ox.distance.nearest_nodes(self.graph, lng2, lat2)
            except Exception as e:
                logger.warning(f"Nearest nodes failed: {e}, using great circle")
                return great_circle((lat1, lng1), (lat2, lng2)).meters
            
            # Check if nodes are the same (points are very close)
            if node1 == node2:
                return great_circle((lat1, lng1), (lat2, lng2)).meters
            
            # Calculate shortest path distance
            try:
                distance_meters = nx.shortest_path_length(
                    self.graph, 
                    node1, 
                    node2, 
                    weight='length'
                )
                return distance_meters
                
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # Fallback to great circle distance if no path found
                return great_circle((lat1, lng1), (lat2, lng2)).meters
                
        except Exception as e:
            logger.error(f"Error calculating walking distance: {e}")
            return great_circle((lat1, lng1), (lat2, lng2)).meters
class ClusterEvaluator:
    """Main class for evaluating clustering performance"""
    
    def __init__(self, user_locations: List[Any], clusters: List[List[Any]], 
                 alpha: float = 1.0, 
                 centroid_method: str = "medoid",
                 graph: Optional[nx.MultiDiGraph] = None,
                 place_name: Optional[str] = None):
        """
        Initialize evaluator with flexible graph handling
        
        Args:
            user_locations: Complete list of all UserLocation objects
            clusters: List of clusters, each cluster is a list of UserLocation objects
            alpha: Weight for destination metrics in combined scores
            centroid_method: Method for finding centroids
            graph: Pre-loaded OSMnx graph (optional)
            place_name: Place name to download graph if not provided
        """
        self.start_time = time.time()
        self.user_locations = user_locations
        self.clusters = clusters
        self.alpha = alpha
        self.centroid_method = centroid_method
        self.unique_labels = list(range(len(clusters)))
        
        # Initialize OSMnx distance calculator
        self.distance_calculator = OSMnxDistanceCalculator(
            graph=graph,
            network_type='walk',
            place_name=place_name
        )
        
        # Create mapping and labels
        self.user_to_cluster, self.labels = self._create_label_mapping()
        self.cluster_indices = self._group_indices_by_cluster()
        
        # Compute distance matrices for both origin and destination
        logger.info("Computing origin distance matrix...")
        self.origin_distance_matrix = self._compute_distance_matrix('origin')
        
        logger.info("Computing destination distance matrix...")
        self.destination_distance_matrix = self._compute_distance_matrix('destination')
        # print(f'\n origin_distance_matrix = {self.origin_distance_matrix}')
        # print(f'\n destination_distance_matrix = {self.destination_distance_matrix}')
        
        # # Find centroids for both origin and destination
        # self.origin_centroids = self._find_cluster_centroids('origin')
        # self.destination_centroids = self._find_cluster_centroids('destination')
        
        # Find centroids using selected method
        if centroid_method == "optimal":
            self.origin_centroids = self._find_cluster_centroids('origin')
            self.destination_centroids = self._find_cluster_centroids('destination')
        elif centroid_method == "approximate":
            self.origin_centroids = self._find_cluster_centroids_approximate('origin')
            self.destination_centroids = self._find_cluster_centroids_approximate('destination')
        else:  # "medoid"
            self.origin_centroids = self._find_cluster_centroids_medoid('origin')
            self.destination_centroids = self._find_cluster_centroids_medoid('destination')
            
        # Compute point-to-centroid distances
        self.origin_point_dists = self._compute_point_to_centroid_distances('origin')
        self.destination_point_dists = self._compute_point_to_centroid_distances('destination')
        # print(f'\n origin_point_dists = {self.origin_point_dists}')
        # print(f'\n destination_point_dists = {self.destination_point_dists}')
        
        self.computation_time = time.time() - self.start_time
        logger.info(f"Cluster evaluation completed in {self.computation_time:.2f} seconds")
    
    def _create_label_mapping(self) -> Tuple[Dict[int, int], np.ndarray]:
        """Create mapping from user_id to cluster label and labels array"""
        user_to_cluster = {}
        labels = []
        
        for cluster_idx, cluster in enumerate(self.clusters):
            for user in cluster:
                user_to_cluster[user.user_id] = cluster_idx
        
        for user in self.user_locations:
            labels.append(user_to_cluster.get(user.user_id, -1))
        
        return user_to_cluster, np.array(labels)
    
    def _group_indices_by_cluster(self) -> Dict[int, List[int]]:
        """Group user indices by their cluster labels"""
        clusters_indices = {}
        for i, user in enumerate(self.user_locations):
            cluster_id = self.user_to_cluster.get(user.user_id, -1)
            if cluster_id != -1:
                if cluster_id not in clusters_indices:
                    clusters_indices[cluster_id] = []
                clusters_indices[cluster_id].append(i)
        return clusters_indices
    
    def _compute_distance_matrix(self, point_type: str) -> np.ndarray:
        """Compute walking distance matrix for origin or destination points"""
        n = len(self.user_locations)
        dist_matrix = np.zeros((n, n))
        
        logger.info(f"Computing {point_type} distance matrix for {n} points...")
        
        for i in range(n):
            user_i = self.user_locations[i]
            dist_matrix[i, i] = 0
            
            for j in range(i + 1, n):
                user_j = self.user_locations[j]
                
                if point_type == 'origin':
                    coords_i = (user_i.origin_lat, user_i.origin_lng)
                    coords_j = (user_j.origin_lat, user_j.origin_lng)
                else:  # destination
                    coords_i = (user_i.destination_lat, user_i.destination_lng)
                    coords_j = (user_j.destination_lat, user_j.destination_lng)
                
                distance_meters = self.distance_calculator.get_walking_distance(
                    coords_i[0], coords_i[1], coords_j[0], coords_j[1], point_type
                )
                
                dist_matrix[i, j] = distance_meters
                dist_matrix[j, i] = distance_meters
        
        logger.info(f"{point_type.capitalize()} distance matrix computation completed")
        return dist_matrix
    
    # def _find_cluster_centroids(self, point_type: str) -> Dict[int, UserLocation]:
    #     """Find centroids using Dijkstra algorithm for origin or destination"""
    #     centroids = {}
    #     distance_matrix = self.origin_distance_matrix if point_type == 'origin' else self.destination_distance_matrix
        
    #     for cluster_id, indices in self.cluster_indices.items():
    #         if not indices:
    #             continue
                
    #         min_total_dist = float('inf')
    #         centroid_idx = -1
            
    #         # Dijkstra-based centroid: point with minimum total distance to others
    #         for i in indices:
    #             total_dist = np.sum(distance_matrix[i, indices])
    #             if total_dist < min_total_dist:
    #                 min_total_dist = total_dist
    #                 centroid_idx = i
            
    #         if centroid_idx != -1:
    #             centroids[cluster_id] = self.user_locations[centroid_idx]
        
    #     logger.info(f"Found {len(centroids)} {point_type} centroids")
    #     return centroids
    #### This function execute the medoid instead of centroid!
    
    def _find_cluster_centroids(self, point_type: str) -> Dict[int, Any]:
        """Find true centroids that minimize total walking distance to cluster points"""
        centroids = {}
        
        for cluster_id, indices in self.cluster_indices.items():
            if not indices:
                continue
                
            if len(indices) == 1:
                # Single point cluster - the point is its own centroid
                centroids[cluster_id] = self.user_locations[indices[0]]
                continue
            
            # Get all points in the cluster
            cluster_points = []
            for idx in indices:
                user = self.user_locations[idx]
                if point_type == 'origin':
                    cluster_points.append((user.origin_lat, user.origin_lng))
                else:
                    cluster_points.append((user.destination_lat, user.destination_lng))
            
            # Find centroid using optimization
            centroid_coords = self._find_optimal_centroid(cluster_points, point_type)
            
            # Find the closest user location to the optimal centroid
            closest_user = self._find_closest_user_to_point(centroid_coords, indices, point_type)
            
            centroids[cluster_id] = closest_user
        
        logger.info(f"Found {len(centroids)} {point_type} centroids")
        return centroids
    
    def _find_optimal_centroid(self, points: List[Tuple[float, float]], point_type: str) -> Tuple[float, float]:
        """Find the optimal centroid coordinates that minimize total walking distance"""
        if len(points) == 1:
            return points[0]
        
        # Use the geographic median as initial guess
        lats, lngs = zip(*points)
        initial_guess = (np.median(lats), np.median(lngs))
        
        # Define objective function: total walking distance from point to all cluster points
        def total_distance(coord):
            total = 0.0
            lat, lng = coord
            for point_lat, point_lng in points:
                dist = self.distance_calculator.get_walking_distance(
                    lat, lng, point_lat, point_lng, point_type
                )
                total += dist
            return total
        
        # Use optimization to find the point that minimizes total distance
        try:
            result = minimize(
                total_distance,
                initial_guess,
                method='L-BFGS-B',
                bounds=[(min(lats), max(lats)), (min(lngs), max(lngs))],
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if result.success:
                return tuple(result.x)
            else:
                # Fallback to geographic median if optimization fails
                return (np.mean(lats), np.mean(lngs))
                
        except Exception as e:
            logger.warning(f"Optimization failed: {e}, using mean coordinates")
            return (np.mean(lats), np.mean(lngs))
    
    def _find_closest_user_to_point(self, target_coords: Tuple[float, float], 
                                  user_indices: List[int], point_type: str) -> Any:
        """Find the user location closest to the optimal centroid coordinates"""
        min_distance = float('inf')
        closest_user = None
        target_lat, target_lng = target_coords
        
        for idx in user_indices:
            user = self.user_locations[idx]
            
            if point_type == 'origin':
                user_coords = (user.origin_lat, user.origin_lng)
            else:
                user_coords = (user.destination_lat, user.destination_lng)
            
            # Calculate distance to target
            distance = self.distance_calculator.get_walking_distance(
                target_lat, target_lng, user_coords[0], user_coords[1], point_type
            )
            
            if distance < min_distance:
                min_distance = distance
                closest_user = user
        
        return closest_user
    
    def _find_cluster_centroids_approximate(self, point_type: str) -> Dict[int, Any]:
        """
        Faster approximate method: use geographic median of coordinates
        then find closest existing point
        """
        centroids = {}
        distance_matrix = self.origin_distance_matrix if point_type == 'origin' else self.destination_distance_matrix
        
        for cluster_id, indices in self.cluster_indices.items():
            if not indices:
                continue
                
            if len(indices) == 1:
                centroids[cluster_id] = self.user_locations[indices[0]]
                continue
            
            # Get coordinates of all points in cluster
            coords = []
            for idx in indices:
                user = self.user_locations[idx]
                if point_type == 'origin':
                    coords.append((user.origin_lat, user.origin_lng))
                else:
                    coords.append((user.destination_lat, user.destination_lng))
            
            # Calculate geographic median (minimizes great circle distance)
            lats, lngs = zip(*coords)
            median_coords = (np.median(lats), np.median(lngs))
            
            # Find closest user to the median
            closest_user = self._find_closest_user_to_point(median_coords, indices, point_type)
            centroids[cluster_id] = closest_user
        
        return centroids
    
    def _find_cluster_centroids_medoid(self, point_type: str) -> Dict[int, Any]:
        """
        Original medoid method: find existing point with minimum total distance
        (kept for comparison or fallback)
        """
        centroids = {}
        distance_matrix = self.origin_distance_matrix if point_type == 'origin' else self.destination_distance_matrix
        
        for cluster_id, indices in self.cluster_indices.items():
            if not indices:
                continue
                
            min_total_dist = float('inf')
            centroid_idx = -1
            
            for i in indices:
                total_dist = np.sum(distance_matrix[i, indices])
                if total_dist < min_total_dist:
                    min_total_dist = total_dist
                    centroid_idx = i
            
            if centroid_idx != -1:
                centroids[cluster_id] = self.user_locations[centroid_idx]
        
        return centroids
    
    def _compute_point_to_centroid_distances(self, point_type: str) -> Dict[int, Dict[int, float]]:
        """Compute distances from points to their cluster centroids"""
        point_dists = {}
        centroids = self.origin_centroids if point_type == 'origin' else self.destination_centroids
        
        for cluster_id, indices in self.cluster_indices.items():
            if cluster_id not in centroids:
                continue
                
            centroid_user = centroids[cluster_id]
            point_dists[cluster_id] = {}
            
            for point_idx in indices:
                point_user = self.user_locations[point_idx]
                
                if point_type == 'origin':
                    point_coords = (point_user.origin_lat, point_user.origin_lng)
                    centroid_coords = (centroid_user.origin_lat, centroid_user.origin_lng)
                else:
                    point_coords = (point_user.destination_lat, point_user.destination_lng)
                    centroid_coords = (centroid_user.destination_lat, centroid_user.destination_lng)
                
                distance_meters = self.distance_calculator.get_walking_distance(
                    point_coords[0], point_coords[1], 
                    centroid_coords[0], centroid_coords[1], 
                    point_type
                )
                
                point_dists[cluster_id][point_idx] = distance_meters
        
        return point_dists
    
    def _calculate_point_type_metrics(self, point_type: str) -> PointTypeMetrics:
        """Calculate metrics for a specific point type (origin/destination)"""
        distance_matrix = self.origin_distance_matrix if point_type == 'origin' else self.destination_distance_matrix
        point_dists = self.origin_point_dists if point_type == 'origin' else self.destination_point_dists
        
        # Calculate SSE
        sse = 0.0
        for cluster_dists in point_dists.values():
            for dist in cluster_dists.values():
                sse += dist ** 2
        
        # Calculate intra-cluster distances
        intra_distances = {}
        for cluster_id, indices in self.cluster_indices.items():
            if len(indices) > 1:
                cluster_dists = distance_matrix[np.ix_(indices, indices)]
                np.fill_diagonal(cluster_dists, np.nan)
                intra_distances[cluster_id] = np.nanmean(cluster_dists)
            else:
                intra_distances[cluster_id] = 0.0
        
        # Calculate radii
        max_radius = 0.0
        total_radius = 0.0
        count = 0
        for cluster_dists in point_dists.values():
            if cluster_dists:
                cluster_max = max(cluster_dists.values())
                cluster_avg = sum(cluster_dists.values()) / len(cluster_dists)
                max_radius = max(max_radius, cluster_max)
                total_radius += cluster_avg
                count += 1
        
        average_radius = total_radius / count if count > 0 else 0.0
        
        # Calculate silhouette score
        valid_mask = self.labels != -1
        valid_labels = self.labels[valid_mask]
        silhouette = -1.0
        
        if len(np.unique(valid_labels)) > 1:
            valid_indices = np.where(valid_mask)[0]
            valid_dist_matrix = distance_matrix[np.ix_(valid_indices, valid_indices)]
            silhouette = silhouette_score(valid_dist_matrix, valid_labels, metric='precomputed')
        
        return PointTypeMetrics(
            sse=sse,
            intra_cluster=np.mean(list(intra_distances.values())) if intra_distances else 0.0,
            max_radius=max_radius,
            average_radius=average_radius,
            silhouette_score=silhouette
        )
    
    def calculate_inter_cluster_distances(self) -> Dict[str, float]:
        """Calculate distances between origin centroids of different clusters"""
        inter_distances = {}
        
        centroid_labels = list(self.origin_centroids.keys())
        for i, label1 in enumerate(centroid_labels):
            for j, label2 in enumerate(centroid_labels[i+1:], i+1):
                centroid1 = self.origin_centroids[label1]
                centroid2 = self.origin_centroids[label2]
                
                distance_meters = self.distance_calculator.get_walking_distance(
                    centroid1.origin_lat, centroid1.origin_lng,
                    centroid2.origin_lat, centroid2.origin_lng,
                    'origin'
                )
                
                key = f"{label1}-{label2}"
                inter_distances[key] = distance_meters
        
        return inter_distances
    
    def calculate_dun_index(self) -> float:
        """Calculate Dunn Index using origin distances"""
        intra_distances = {}
        for cluster_id, indices in self.cluster_indices.items():
            if len(indices) > 1:
                cluster_dists = self.origin_distance_matrix[np.ix_(indices, indices)]
                np.fill_diagonal(cluster_dists, np.nan)
                intra_distances[cluster_id] = np.nanmean(cluster_dists)
            else:
                intra_distances[cluster_id] = 0.0
        
        inter_distances = self.calculate_inter_cluster_distances()
        
        if not intra_distances or not inter_distances:
            return 0.0
        
        max_intra = max(intra_distances.values())
        min_inter = min(inter_distances.values())
        
        return min_inter / max_intra if max_intra > 0 else float('inf')
    
    def evaluate(self) -> ClusterMetrics:
        """Run all evaluations and return comprehensive metrics"""
        # Calculate metrics for both point types
        origin_metrics = self._calculate_point_type_metrics('origin')
        destination_metrics = self._calculate_point_type_metrics('destination')
        
        # Calculate combined metrics
        combined_sse = origin_metrics.sse + self.alpha * destination_metrics.sse
        combined_intra = origin_metrics.intra_cluster + self.alpha * destination_metrics.intra_cluster
        combined_silhouette = (origin_metrics.silhouette_score + self.alpha * destination_metrics.silhouette_score) / (1 + self.alpha)
        
        cluster_sizes = {
            cluster_id: len(indices) for cluster_id, indices in self.cluster_indices.items()
        }
        
        return ClusterMetrics(
            combined_sse=combined_sse,
            combined_intra_cluster=combined_intra,
            combined_silhouette=combined_silhouette,
            dun_index=self.calculate_dun_index(),
            origin_metrics=origin_metrics,
            destination_metrics=destination_metrics,
            cluster_sizes=cluster_sizes,
            inter_cluster_distances=self.calculate_inter_cluster_distances(),
            origin_centroids=self.origin_centroids,
            destination_centroids=self.destination_centroids,
            computation_time=self.computation_time,
            alpha=self.alpha
        )

# Main function for easy access
def evaluate_user_clustering(user_locations: List[Any], 
                           clusters: List[List[Any]],
                           alpha: float = 1.0,
                           centroid_method: str = "medoid",
                           graph: Optional[nx.MultiDiGraph] = None,
                           place_name: Optional[str] = "Savojbolagh County, Alborz Province, Iran") -> ClusterMetrics:
    """
    Evaluate clustering with flexible graph handling
    
    Args:
        user_locations: Complete list of all UserLocation objects
        clusters: List of clusters, each cluster is a list of UserLocation objects
        alpha: Weight for destination metrics in combined scores
        centroid_method: Method for finding centroids
        graph: Pre-loaded OSMnx graph (optional)
        place_name: Place name to download graph if not provided
    
    Returns:
        Comprehensive clustering metrics
    """
    evaluator = ClusterEvaluator(
        user_locations=user_locations,
        clusters=clusters,
        alpha=alpha,
        centroid_method=centroid_method,
        graph=graph,
        place_name=place_name
    )
    return evaluator.evaluate()
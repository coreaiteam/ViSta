from typing import List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np
import osmnx as ox
import networkx as nx
import time
from sklearn.neighbors import BallTree
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import math

from ..models import UserLocation, ClusterGroup
from .monitoring import AdvancedResourceMonitor


class ClusteringEngine:
    def __init__(self, places: List[str] = ["Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran"],
                 k_nearest: int = 100, similarity_threshold: float = 0.7,
                ):
        """
        Initialize the spatial clustering system.

        Args:
            places: Locations to extract graph from OSM
            k_nearest: Number of nearest nodes to consider for each location
            similarity_threshold: Minimum similarity threshold for clustering
            cache_file: File to cache precomputed data
        """
        self.engine_name = "MIPS_Engine"
        self.places = places
        self.k_nearest = k_nearest
        self.similarity_threshold = similarity_threshold
        self._make_cache_file_name(places=places)

        # Graph and precomputed data
        self.G = None
        self.nodes_list = None
        self.node_to_idx = None
        self.nearest_nodes_cache = {}
        self.distance_cache = {}

        # BallTree for fast nearest neighbor search
        self.ball_tree = None
        self.node_coords = None

        # Initialize the system
        self._load_or_compute_graph()
        self._build_ball_tree()
        self._load_or_compute_precomputed_data()

    def _make_cache_file_name(self, places: List[str]):
        self.cache_file = "MIPS_"

        for p in places:
            self.cache_file += p.replace(", ", "-").replace(" ", "_")
        self.cache_file += ".pkl"

    def _load_or_compute_graph(self):
        """Load OSM graph for the specified places."""
        print(f"Loading graph for {self.places}...")
        self.G = ox.graph_from_place(self.places, network_type='walk')
        self.nodes_list = list(self.G.nodes())
        self.node_to_idx = {node: idx for idx,
                            node in enumerate(self.nodes_list)}
        print(f"Graph loaded with {len(self.nodes_list)} nodes")

    def _build_ball_tree(self):
        """Build BallTree for fast nearest neighbor search."""
        print("Building BallTree for fast nearest neighbor search...")

        # Extract coordinates for all nodes in radians (required for haversine distance)
        self.node_coords = []
        for node in self.nodes_list:
            lat = np.radians(self.G.nodes[node]['y'])
            lng = np.radians(self.G.nodes[node]['x'])
            self.node_coords.append([lat, lng])

        self.node_coords = np.array(self.node_coords)

        # Build BallTree with haversine metric (great for geographic coordinates)
        self.ball_tree = BallTree(self.node_coords, metric='haversine')
        print("BallTree built successfully")

    def _load_or_compute_precomputed_data(self):
        """Load precomputed data or compute if doesn't exist."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.nearest_nodes_cache = cache_data['nearest_nodes']
                self.distance_cache = cache_data['distances']
                print("Loaded precomputed data from cache")
        except FileNotFoundError:
            print("Cache not found. Computing precomputed data...")
            self._precompute_data()
            self._save_cache()

    def _precompute_data(self):
        """Precompute nearest nodes and distances for optimization."""
        print("Precomputing nearest nodes and distances...")

        # For each node, find 200 nearest nodes and their distances
        for i, node in enumerate(self.nodes_list):
            if i % 100 == 0:
                print(f"Processing node {i}/{len(self.nodes_list)}")

            try:
                # Get distances to all other nodes (limited by network connectivity)
                distances = nx.single_source_dijkstra_path_length(
                    self.G, node, cutoff=5000, weight='length'  # 5km cutoff
                )

                # Sort by distance and take top 200
                sorted_distances = sorted(
                    distances.items(), key=lambda x: x[1])[:200]

                # Store nearest nodes
                nearest_nodes = [(target_node, dist)
                                 for target_node, dist in sorted_distances]
                self.nearest_nodes_cache[node] = nearest_nodes

                # Store distance matrix for this node
                self.distance_cache[node] = {
                    target: dist for target, dist in sorted_distances}

            except Exception as e:
                print(f"Error processing node {node}: {e}")
                # Fallback to Euclidean distance
                node_coords = (self.G.nodes[node]
                               ['y'], self.G.nodes[node]['x'])
                euclidean_distances = []

                for other_node in self.nodes_list:
                    if other_node != node:
                        other_coords = (
                            self.G.nodes[other_node]['y'], self.G.nodes[other_node]['x'])
                        dist = ox.distance.euclidean_dist_vec(
                            node_coords[0], node_coords[1],
                            other_coords[0], other_coords[1]
                        )
                        euclidean_distances.append((other_node, dist))

                euclidean_distances.sort(key=lambda x: x[1])
                self.nearest_nodes_cache[node] = euclidean_distances[:200]
                self.distance_cache[node] = {
                    target: dist for target, dist in euclidean_distances[:200]}

    def _save_cache(self):
        """Save precomputed data to cache file."""
        cache_data = {
            'nearest_nodes': self.nearest_nodes_cache,
            'distances': self.distance_cache
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Precomputed data saved to cache")

    def _find_k_nearest_nodes(self, coords: Tuple[float, float]) -> List[Tuple[int, float]]:
        """Find k nearest nodes to given coordinates using BallTree."""
        lat, lng = coords

        # Convert coordinates to radians for BallTree query
        query_point = np.array([[np.radians(lat), np.radians(lng)]])

        # Find nearest node using BallTree (much faster than ox.distance.nearest_nodes)
        distances, indices = self.ball_tree.query(query_point, k=1)
        nearest_node_idx = indices[0][0]
        nearest_node = self.nodes_list[nearest_node_idx]

        # Get precomputed nearest nodes
        if nearest_node in self.nearest_nodes_cache:
            candidates = self.nearest_nodes_cache[nearest_node]
        else:
            # Fallback to BallTree for finding k nearest nodes directly
            distances, indices = self.ball_tree.query(
                query_point, k=min(self.k_nearest, len(self.nodes_list)))
            candidates = []

            for i, idx in enumerate(indices[0]):
                node = self.nodes_list[idx]
                # Convert distance from radians back to meters (approximate)
                dist_meters = distances[0][i] * \
                    6371000  # Earth's radius in meters
                candidates.append((node, dist_meters))

        # Return top k candidates
        return candidates[:self.k_nearest]

    def _normalize_distances(self, distances: List[float], sigma: float = 500.0) -> List[float]:
        """Convert distances to similarity weights using a Gaussian kernel.

        Args:
            distances: list of distances in meters
            sigma: scale parameter controlling decay rate (meters)
        """
        if not distances:
            return []

        # Gaussian kernel: sim = exp(-(d^2) / (2*sigma^2))
        similarities = [math.exp(-(d ** 2) / (2 * sigma ** 2))
                        for d in distances]

        # Keep magnitudes — no sum-to-1 normalization
        return similarities

    def _create_feature_matrix(self, user_locations: List[UserLocation]) -> np.ndarray:
        """Create the bipartite feature matrix for users."""
        n_nodes = len(self.nodes_list)
        n_users = len(user_locations)

        # Matrix with 2n columns (n for origins, n for destinations)
        matrix = np.zeros((n_users, 2 * n_nodes))

        for user_idx, user in enumerate(user_locations):
            # Process origin
            origin_nearest = self._find_k_nearest_nodes(user.origin_coords)
            origin_distances = [dist for _, dist in origin_nearest]
            origin_weights = self._normalize_distances(origin_distances)

            for (node, _), weight in zip(origin_nearest, origin_weights):
                node_idx = self.node_to_idx[node]
                matrix[user_idx, node_idx] = weight  # Origin columns

            # Process destination
            dest_nearest = self._find_k_nearest_nodes(user.destination_coords)
            dest_distances = [dist for _, dist in dest_nearest]
            dest_weights = self._normalize_distances(dest_distances)

            for (node, _), weight in zip(dest_nearest, dest_weights):
                node_idx = self.node_to_idx[node]
                # Destination columns
                matrix[user_idx, n_nodes + node_idx] = weight

        return matrix

    def _calculate_meeting_points(self, users: List[UserLocation]) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """Calculate meeting points for origin and destination."""
        if not users:
            return None, None

        # Calculate centroid for origins
        origin_lats = [u.origin_lat for u in users]
        origin_lngs = [u.origin_lng for u in users]
        origin_meeting = (np.mean(origin_lats), np.mean(origin_lngs))

        # Calculate centroid for destinations
        dest_lats = [u.destination_lat for u in users]
        dest_lngs = [u.destination_lng for u in users]
        dest_meeting = (np.mean(dest_lats), np.mean(dest_lngs))

        return origin_meeting, dest_meeting

    def cluster_users(self, user_locations: List[UserLocation]) -> List[ClusterGroup]:
        """
        Main clustering function that groups similar users.

        Args:
            user_locations: List of UserLocation objects

        Returns:
            List of ClusterGroup objects
        """
        with AdvancedResourceMonitor() as monitor:
            if len(user_locations) < 2:
                return []

            print(f"Clustering {len(user_locations)} users...")

            # Create feature matrix

            feature_matrix = self._create_feature_matrix(user_locations)

            # Calculate similarity matrix using cosine similarity (approximates MIPS for normalized vectors)
            similarity_matrix = cosine_similarity(feature_matrix)

            # Find groups
            groups = []
            used_users = set()

            for i, user1 in enumerate(user_locations):
                if user1.user_id in used_users:
                    continue

                # Find most similar users
                similarities = similarity_matrix[i]
                similar_indices = []

                for j, sim in enumerate(similarities):
                    if i != j and sim >= self.similarity_threshold and user_locations[j].user_id not in used_users:
                        similar_indices.append((j, sim))

                # Sort by similarity
                similar_indices.sort(key=lambda x: x[1], reverse=True)

                # Form groups of up to 3 users
                if similar_indices:
                    group_users = [user1]
                    used_users.add(user1.user_id)

                    # Add up to 2 more users
                    for j, sim in similar_indices[:2]:
                        if user_locations[j].user_id not in used_users:
                            group_users.append(user_locations[j])
                            used_users.add(user_locations[j].user_id)

                    # Calculate meeting points
                    origin_meeting, dest_meeting = self._calculate_meeting_points(
                        group_users)

                    # Create group
                    group_id = f"group_{'_'.join(str(u.user_id) for u in group_users)}_{int(time.time())}"
                    group = ClusterGroup(
                        group_id=group_id,
                        users=group_users,
                        created_at=datetime.now(timezone.utc),
                        meeting_point_origin=origin_meeting,
                        meeting_point_destination=dest_meeting,
                        status="complete" if len(group_users) == 3 else "forming"
                    )
                    groups.append(group)
            return groups

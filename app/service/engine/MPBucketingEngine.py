import os
import uuid
import math
import pickle
import tempfile
import traceback

import numpy as np
import osmnx as ox
import networkx as nx

from collections import defaultdict
from sklearn.neighbors import BallTree
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

from ..models import UserLocation, ClusterGroup
from .monitoring import AdvancedResourceMonitor

@dataclass
class InternalClusterGroup:
    """A dataclass to represent a group of users clustered together."""
    group_id: str = field(default_factory=lambda: str(
        uuid.uuid4()))  # Unique identifier for the group
    users: List[UserLocation] = field(
        default_factory=list)  # List of users in the group
    created_at: datetime = field(default_factory=lambda: datetime.now(
        timezone.utc))  # Timestamp of group creation
    # Coordinates of the origin meeting point
    meeting_point_origin: Optional[Tuple[float, float]] = None
    # Ascent: 0
    meeting_point_destination: Optional[Tuple[float, float]] = None
    orig_buckets: Dict[int, float] = field(default_factory=dict)  # nearest nodes to origin node
    dest_buckets: Dict[int, float] = field(default_factory=dict)  # nearest nodes to destination node

    @property
    def status(self) -> str:
        return "complete" if len(self.users) == 3 else "forming" if len(self.users) == 2 else "expired"

    @property
    def is_complete(self) -> bool:
        return len(self.users) >= 3

    def __eq__(self, other):
        """Compare two InternalClusterGroup instances for equality based on group_id."""
        if not isinstance(other, InternalClusterGroup):
            return False
        return self.group_id == other.group_id

    def __hash__(self):
        """Generate hash based on group_id."""
        return hash(self.group_id)

    def update(self, new_user_location: UserLocation, remove=False) -> Optional[int]:
        """
        Update the group by adding or removing a user and recalculate meeting points.

        Args:
            new_user_location (UserLocation): User location to add or remove.
            remove (bool): If True, remove the user; otherwise, add them.

        Returns:
            Optional[int]: ID of the user left alone if group expires, else None.
        """
        if remove:
            if new_user_location in self.users:
                self.users.remove(new_user_location)
            if len(self.users) == 1:
                return self.users[0].user_id
        else:
            self.users.append(new_user_location)  # Add new user to the group
        return None

    def to_cluster_group(self) -> ClusterGroup:
        """
        Convert InternalClusterGroup to ClusterGroup for external use.

        Returns:
            ClusterGroup: A ClusterGroup instance with the same attributes.
        """
        return ClusterGroup(
            group_id=self.group_id,
            users=self.users,
            created_at=self.created_at,
            meeting_point_origin=self.meeting_point_origin,
            meeting_point_destination=self.meeting_point_destination,
            status=self.status
        )


@dataclass
class User:
    """A dataclass to represent a user with location and group information."""
    user_id: int  # Unique identifier for the user
    user_location: UserLocation  # User's location data
    group: Optional[InternalClusterGroup] = None  # Group the user belongs to, if any
    orig_buckets: Dict[int, float] = field(default_factory=dict)  # k nearest nodes to origin node
    dest_buckets: Dict[int, float] = field(default_factory=dict)  # k nearest nodes to destination node
    orig_node: int = -1  # nearest node to origin location
    dest_node: int = -1  # nearest node to destination location

    @property
    def origin_coords(self) -> Tuple[float, float]:
        """Get the user's origin coordinates."""
        return (self.user_location.origin_lat, self.user_location.origin_lng)

    @property
    def destination_coords(self) -> Tuple[float, float]:
        """Get the user's destination coordinates."""
        return (self.user_location.destination_lat, self.user_location.destination_lng)

    @property
    def companions_number(self) -> int:
        """Get the number of companions in the user's group."""
        if self.group:
            return len(self.group.users) - 1  # Exclude the user themselves
        return 0

    def update_group(self, group: Optional[InternalClusterGroup]):
        """
        Update the user's group assignment.

        Args:
            group (Optional[InternalClusterGroup]): The group to assign the user to.
        """
        self.group = group

    def to_dict(self) -> Dict:
        """
        Convert user data to a dictionary.

        Returns:
            Dict: An empty dictionary (to be implemented as needed).
        """
        return {}


class ClusteringEngine:
    """A class to cluster users based on their origin and destination locations."""

    def __init__(self, places: List[str] = ["Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran"],
                 k_nearest: int = 100, similarity_threshold: float = 0.5,
                ):
        """
        Initialize the clustering engine with geographical and clustering parameters.

        Args:
            places List[str]: List of geographical areas for the street network.
            k_nearest (int): Number of nearest nodes to consider for signatures.
            similarity_threshold (float): Minimum similarity for clustering users.
        """
        self.engine_name = "MPBucketing_Engine"
        self.places = places
        self.k_nearest = k_nearest
        self.similarity_threshold = similarity_threshold
        self._make_cache_file_name(places=places)
        self.G = None  # Street network graph
        self.nodes_list = None  # List of graph nodes
        self.node_to_idx = None  # Mapping of nodes to indices
        # Cache for precomputed nearest nodes
        self.nearest_nodes_cache: Dict[int, Dict[int, float]] = {}
        self.ball_tree = None  # BallTree for nearest neighbor search
        self.node_coords = None  # Coordinates of graph nodes
        self.orig_buckets: Dict[int, Dict[int, Tuple[List[int], float]]] = defaultdict(dict)
        self.dest_buckets: Dict[int, Dict[int, Tuple[List[int], float]]] = defaultdict(dict)
        self.users: Dict[int, User] = {}  # Dictionary of users
        self._load_or_compute_graph()  # Load street network graph
        self._build_ball_tree()  # Build BallTree for spatial queries
        self._load_or_compute_precomputed_data()

    def _make_cache_file_name(self, places: List[str]):
        """Generate cache file name based on places."""
        self.cache_file = "MPB_"
        for p in places:
            self.cache_file += p.replace(", ", "-").replace(" ", "_")
        self.cache_file += ".pkl"
        # Directory for temporary chunk files
        self.temp_dir = os.path.join(os.path.dirname(self.cache_file), "temp_chunks")
        os.makedirs(self.temp_dir, exist_ok=True)

    def _load_or_compute_graph(self):
        """Load the street network graph from OpenStreetMap."""
        print(f"Loading graph for {self.places}...")
        self.G = ox.graph_from_place(self.places, network_type='walk')
        self.nodes_list = list(self.G.nodes())  # Get list of graph nodes
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        print(f"Graph loaded with {len(self.nodes_list)} nodes")

    def _build_ball_tree(self):
        """Build a BallTree for efficient nearest neighbor searches."""
        print("Building BallTree for fast nearest neighbor search...")
        self.node_coords = []
        for node in self.nodes_list:
            # Convert latitude to radians
            lat = np.radians(self.G.nodes[node]['y'])
            # Convert longitude to radians
            lng = np.radians(self.G.nodes[node]['x'])
            self.node_coords.append([lat, lng])
        self.node_coords = np.array(self.node_coords)
        # Use haversine metric for geographical data
        self.ball_tree = BallTree(self.node_coords, metric='haversine')
        print("BallTree built successfully")

    def _load_or_compute_precomputed_data(self):
        """Load precomputed data with memory safety and resume capability."""
        # Check if complete cache file exists
        if os.path.exists(self.cache_file):
            try:
                file_size = os.path.getsize(self.cache_file) / (1024 * 1024)  # MB
                print(f"Cache file size: {file_size:.2f} MB")
                
                with open(self.cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.nearest_nodes_cache = cache_data['nearest_nodes']
                    print(f"Loaded {len(self.nearest_nodes_cache)} nodes from cache")
                return
            except Exception as e:
                print(f"Error loading cache file: {e}. Recomputing...")
        
        # Check for existing temporary chunk files from previous run
        temp_files = self._find_existing_chunk_files()
        
        if temp_files:
            print(f"Found {len(temp_files)} incomplete chunk files from previous run. Resuming computation...")
            self._resume_from_chunk_files(temp_files)
        else:
            print("No cache found. Starting fresh computation...")
            self._precompute_data()

    def _find_existing_chunk_files(self) -> List[str]:
        """Find existing temporary chunk files from previous runs."""
        chunk_files = []
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                if file.startswith("chunk_") and file.endswith(".pkl"):
                    chunk_files.append(os.path.join(self.temp_dir, file))
        
        # Sort files by chunk start index
        chunk_files.sort(key=lambda x: int(x.split('_')[-2]))
        return chunk_files

    def _resume_from_chunk_files(self, temp_files: List[str]):
        """Resume computation from existing chunk files."""
        # First, load all completed chunks into memory
        processed_nodes = set()
        for temp_file in temp_files:
            try:
                with open(temp_file, 'rb') as f:
                    chunk_data = pickle.load(f)
                    self.nearest_nodes_cache.update(chunk_data)
                    processed_nodes.update(chunk_data.keys())
                print(f"Loaded chunk {temp_file} with {len(chunk_data)} nodes")
            except Exception as e:
                print(f"Error loading chunk file {temp_file}: {e}")
        
        # Find which nodes still need to be processed
        all_nodes = set(self.nodes_list)
        nodes_to_process = list(all_nodes - processed_nodes)
        
        if not nodes_to_process:
            print("All nodes already processed. Merging into final cache...")
            self._create_final_cache_from_chunks(temp_files)
            return
        
        print(f"Resuming computation: {len(processed_nodes)} nodes already processed, "
              f"{len(nodes_to_process)} nodes remaining")
        
        # Continue processing remaining nodes
        self._process_remaining_nodes(nodes_to_process, temp_files)

    def _process_remaining_nodes(self, nodes_to_process: List[int], existing_temp_files: List[str]):
        """Process the remaining nodes and create new chunk files."""
        total_remaining = len(nodes_to_process)
        chunk_size = 5000
        
        for chunk_start in range(0, total_remaining, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_remaining)
            chunk_nodes = nodes_to_process[chunk_start:chunk_end]
            
            chunk_data = {}
            for i, node in enumerate(chunk_nodes):
                if i % 100 == 0:
                    print(f"Processing remaining node {chunk_start + i}/{total_remaining}")
                
                try:
                    # Calculate shortest path distances
                    distances = nx.single_source_dijkstra_path_length(
                        self.G, node, cutoff=5000, weight='length'
                    )
                    sorted_distances = sorted(distances.items(), key=lambda x: x[1])[:self.k_nearest]
                    chunk_data[node] = {target_node: dist for target_node, dist in sorted_distances}
                    
                except Exception as e:
                    print(f"Error processing node {node}: {e}")
                    # Fallback to Euclidean distance
                    node_coords = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                    euclidean_distances = []
                    for other_node in self.nodes_list:
                        if other_node != node:
                            other_coords = (self.G.nodes[other_node]['y'], self.G.nodes[other_node]['x'])
                            dist = ox.distance.euclidean_dist_vec(
                                node_coords[0], node_coords[1],
                                other_coords[0], other_coords[1]
                            )
                            euclidean_distances.append((other_node, dist))
                    euclidean_distances.sort(key=lambda x: x[1])
                    chunk_data[node] = dict(euclidean_distances[:self.k_nearest])
            
            # Save this chunk to temporary file
            temp_file = os.path.join(self.temp_dir, f"chunk_{chunk_start}_{chunk_end}.pkl")
            with open(temp_file, 'wb') as f:
                pickle.dump(chunk_data, f)
            existing_temp_files.append(temp_file)
            print(f"Saved new chunk with {len(chunk_data)} nodes")
            
            # Update in-memory cache
            self.nearest_nodes_cache.update(chunk_data)
            
            # Free memory
            del chunk_data
        
        # Create final cache from all chunk files
        self._create_final_cache_from_chunks(existing_temp_files)

    def _precompute_data(self):
        """Precompute nearest nodes and distances with guaranteed memory safety."""
        print("Precomputing nearest nodes and distances...")
        total_nodes = len(self.nodes_list)
        
        temp_files = []
        
        try:
            # Process in small chunks to avoid memory issues
            chunk_size = 5000
            for chunk_start in range(0, total_nodes, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_nodes)
                chunk_nodes = self.nodes_list[chunk_start:chunk_end]
                
                chunk_data = {}
                for i, node in enumerate(chunk_nodes):
                    if i % 100 == 0:
                        print(f"Processing node {chunk_start + i}/{total_nodes}")
                    
                    try:
                        # Calculate shortest path distances
                        distances = nx.single_source_dijkstra_path_length(
                            self.G, node, cutoff=5000, weight='length'
                        )
                        sorted_distances = sorted(distances.items(), key=lambda x: x[1])[:self.k_nearest]
                        chunk_data[node] = {target_node: dist for target_node, dist in sorted_distances}
                        
                    except Exception as e:
                        print(f"Error processing node {node}: {e}")
                        # Fallback to Euclidean distance
                        node_coords = (self.G.nodes[node]['y'], self.G.nodes[node]['x'])
                        euclidean_distances = []
                        for other_node in self.nodes_list:
                            if other_node != node:
                                other_coords = (self.G.nodes[other_node]['y'], self.G.nodes[other_node]['x'])
                                dist = ox.distance.euclidean_dist_vec(
                                    node_coords[0], node_coords[1],
                                    other_coords[0], other_coords[1]
                                )
                                euclidean_distances.append((other_node, dist))
                        euclidean_distances.sort(key=lambda x: x[1])
                        chunk_data[node] = dict(euclidean_distances[:self.k_nearest])
                
                # Save this chunk to temporary file
                temp_file = os.path.join(self.temp_dir, f"chunk_{chunk_start}_{chunk_end}.pkl")
                with open(temp_file, 'wb') as f:
                    pickle.dump(chunk_data, f)
                temp_files.append(temp_file)
                print(f"Saved chunk {len(temp_files)} - {len(chunk_data)} nodes")
                
                # Update in-memory cache
                self.nearest_nodes_cache.update(chunk_data)
                
                # Free memory
                del chunk_data
            
            # Create final cache from all chunk files
            self._create_final_cache_from_chunks(temp_files)
            
        finally:
            # Clean up temporary files after successful completion
            if hasattr(self, 'nearest_nodes_cache') and len(self.nearest_nodes_cache) == total_nodes:
                self._cleanup_temp_files()

    def _create_final_cache_from_chunks(self, temp_files: List[str]):
        """Create final cache file from all chunk files."""
        print("Creating final cache file from chunks...")
        
        # Final cache structure
        final_cache = {'nearest_nodes': self.nearest_nodes_cache}
        
        # Save to final cache file
        with open(self.cache_file, 'wb') as f:
            pickle.dump(final_cache, f)
        
        print(f"Final cache created with {len(self.nearest_nodes_cache)} nodes")
        
        # Clean up temporary files
        self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        """Clean up temporary chunk files."""
        if os.path.exists(self.temp_dir):
            for file in os.listdir(self.temp_dir):
                if file.startswith("chunk_") and file.endswith(".pkl"):
                    try:
                        os.remove(os.path.join(self.temp_dir, file))
                    except:
                        pass
            print("Temporary chunk files cleaned up")

    def _get_nearest_nodes(self, coords: Tuple[float, float]) -> Tuple[Dict[int, float], int]:
        lat, lng = coords
        query_point = np.array([[np.radians(lat), np.radians(lng)]])
        _, indices = self.ball_tree.query(
            query_point, k=1)  # Find nearest node
        nearest_node_idx = indices[0][0]
        nearest_node = self.nodes_list[nearest_node_idx]
        return self.nearest_nodes_cache[nearest_node], nearest_node

    def _collect_candidate_users(
        self,
        orig_user_buckets: Dict[int, float],
        dest_user_buckets: Dict[int, float],
    ) -> Dict[int, dict]:

        """ Final dictionary: {
                candid_id:
                {
                    "orig": [(bucket_id, dist)],
                    "dest": [(bucket_id, dist)],
                    "num": number_of_users,
                    "user_ids": [user_ids]
                }
            }
        """
        candidates = defaultdict(lambda: {"orig": [], "dest": [], "num": 0, "user_ids": []})

        # Collect users from origin buckets
        for bucket_id, _ in orig_user_buckets.items():
            if bucket_id in self.orig_buckets:
                for cid, (user_ids, dist) in self.orig_buckets[bucket_id].items():
                    candidates[cid]["orig"].append((bucket_id, dist))
                    candidates[cid]["user_ids"].extend(user_ids)
                    candidates[cid]["num"] = len(set(candidates[cid]["user_ids"]))

        # Collect users from destination buckets
        for bucket_id, _ in dest_user_buckets.items():
            if bucket_id in self.dest_buckets:
                for cid, (user_ids, dist) in self.dest_buckets[bucket_id].items():
                    candidates[cid]["dest"].append((bucket_id, dist))
                    # Also ensure we have the user_ids for destination
                    if cid not in candidates:
                        candidates[cid] = {"orig": [], "dest": [], "num": 0, "user_ids": []}
                    candidates[cid]["user_ids"].extend(user_ids)
                    candidates[cid]["num"] = len(set(candidates[cid]["user_ids"]))

        # Remove duplicates in user_ids and update count
        for cid, info in candidates.items():
            info["user_ids"] = list(set(info["user_ids"]))
            info["num"] = len(info["user_ids"])

        # Keep only users that exist in both origin and destination buckets
        return {cid: info for cid, info in candidates.items() if info["orig"] and info["dest"]}
    
    def _get_coords(self, node_id: int) -> Tuple[float, float]:
        """
        Get geographical coordinates for a given node ID.
        
        Args:
            node_id (int): The ID of the node in the graph.
            
        Returns:
            Tuple[float, float]: (latitude, longitude) coordinates of the node.
        """
        if node_id in self.G.nodes:
            node_data = self.G.nodes[node_id]
            return (np.float64(node_data['y']), np.float64(node_data['x']))  # (latitude, longitude)
    
    def _form_group(
        self,
        main_user: User,
        candidates: Dict[int, Dict[str, List[Tuple[int, float]]]],
        threshold: float = None,
    ) -> Optional[InternalClusterGroup]:
        """
        Assign the best group (co-travelers) for the given user.

        Args:
            main_user: The user we want to group
            candidates: {
                candid_id:
                {
                    "orig": [(bucket_id, dist)],
                    "dest": [(bucket_id, dist)],
                    "num": number_of_users,
                    "user_ids": [user_ids]
                }
            }
            threshold: max allowed distance

        Returns:
            InternalClusterGroup assigned to the user
        """
        if threshold is None:
            threshold = self.similarity_threshold

        scored_candidates = []

        for cand_id, info in candidates.items():
            num = info["num"] + 1
            # --- find best origin bucket ---
            best_orig_bucket, min_orig_dist = min(
                info["orig"], key=lambda x: math.sqrt(main_user.orig_buckets[x[0]]**2 + x[1]**2) / num
            )
            min_orig_dist = math.sqrt(main_user.orig_buckets[best_orig_bucket]**2 +
                                    dict(info["orig"])[best_orig_bucket]**2) / num

            # --- find best destination bucket ---
            best_dest_bucket, min_dest_dist = min(
                info["dest"], key=lambda x: math.sqrt(main_user.dest_buckets[x[0]]**2 + x[1]**2) / num
            )
            min_dest_dist = math.sqrt(main_user.dest_buckets[best_dest_bucket]**2 +
                                    dict(info["dest"])[best_dest_bucket]**2) / num

            total_dist = min_orig_dist + min_dest_dist

            if math.exp(-total_dist / 1000) >= threshold:
                scored_candidates.append(
                    (cand_id, info["user_ids"], total_dist, best_orig_bucket, best_dest_bucket)
                )

        if len(scored_candidates) == 0:
            return None

        # select best candidate based on total_dist (lowest)
        selected = min(scored_candidates, key=lambda x: x[2])


        members = [main_user] + [self.users[uid] for uid in selected[1] if uid in self.users]

        if len(members) == 3:
            group = members[-1].group
            group.update(main_user.user_location)
        elif len(members) == 2:
            # create group
            group = InternalClusterGroup(
                users=[u.user_location for u in members],
                created_at=datetime.now(timezone.utc),
            )
            members[1].update_group(group=group)

        main_user.update_group(group=group)
        group.meeting_point_origin = self._get_coords(selected[3])  # todo
        group.meeting_point_destination = self._get_coords(selected[4]) # todo

        # --- update bucket structures ---
        if group.is_complete:
            for b in group.orig_buckets.keys():
                if b in self.orig_buckets and group.group_id in self.orig_buckets[b]:
                    del self.orig_buckets[b][group.group_id]
            for b in group.dest_buckets.keys():
                if b in self.dest_buckets and group.group_id in self.dest_buckets[b]:
                    del self.dest_buckets[b][group.group_id]
            return group

        # 1. remove individual users
        user = members[1]
        for b in user.orig_buckets:
            if b in self.orig_buckets and user.user_id in self.orig_buckets[b]:
                del self.orig_buckets[b][user.user_id]
        for b in user.dest_buckets:
            if b in self.dest_buckets and user.user_id in self.dest_buckets[b]:
                del self.dest_buckets[b][user.user_id]


        # 2. add group to all common buckets
        cand_id = selected[0]
        orig_buckets = candidates[cand_id]["orig"]
        dest_buckets = candidates[cand_id]["dest"]
        
        group_id = group.group_id
        group.orig_buckets = {b:d for (b, d) in orig_buckets}
        group.dest_buckets = {b:d for (b, d) in dest_buckets}


        for b, dist in orig_buckets:
            new_dist = math.sqrt(main_user.orig_buckets[b]**2 + dist**2)
            self.orig_buckets[b][group_id] = ([u.user_id for u in members], new_dist)

        for b, dist in dest_buckets:
            new_dist = math.sqrt(main_user.dest_buckets[b]**2 + dist**2)
            self.dest_buckets[b][group_id] = ([u.user_id for u in members], new_dist)

        return group

    def _create_user(self, user_location: UserLocation, orig_buckets: Dict[int, float],
                     dest_buckets: Dict[int, float], orig_node: int, dest_node: int) -> User:

        new_user = User(
            user_id=user_location.user_id,
            user_location=user_location,
            group=None,
            orig_buckets=orig_buckets,
            dest_buckets=dest_buckets,
            orig_node=orig_node,
            dest_node=dest_node,
        )
        self.users[new_user.user_id] = new_user
        return new_user

    def _add_user_to_buckets(self, user: User):
        """Add a user to their corresponding buckets."""
        for b, dist in user.orig_buckets.items():
            self.orig_buckets[b][user.user_id] = ([user.user_id], dist)
        for b, dist in user.dest_buckets.items():
            self.dest_buckets[b][user.user_id] = ([user.user_id], dist)

    def _update_group_buckets(self, group: InternalClusterGroup):
        """
        Update the buckets for a group after changes in its membership.
        Only updates if the group is not complete.
        """
        
        # Get current group members
        user_ids_in_group = [u.user_id for u in group.users]
        users_in_group = [self.users[uid] for uid in user_ids_in_group]

        u1, u2 = users_in_group[0:1]

        # Find common origin buckets with minimum distances
        common_orig_buckets = {}
        for b, dist1 in u1.orig_buckets.items():
            if b in u2.orig_buckets.keys():
                dist2 = u2.orig_buckets[b]
                common_orig_buckets[b] = math.sqrt(dist1**2 + dist2**2) / 2

        # Find common destination buckets with minimum distances
        common_dest_buckets = {}
        for b, dist1 in u1.dest_buckets.items():
            if b in u2.dest_buckets.keys():
                dist2 = u2.dest_buckets[b]
                common_dest_buckets[b] = math.sqrt(dist1**2 + dist2**2) / 2

        # Add group to buckets
        for b, dist in common_orig_buckets.items():
            self.orig_buckets[b][group.group_id] = (user_ids_in_group, dist)

        for b, dist in common_dest_buckets.items():
            self.dest_buckets[b][group.group_id] = (user_ids_in_group, dist)

    def add_user(self, user_location: UserLocation) -> Optional[InternalClusterGroup]:
        """
        Add a user to the clustering engine and attempt to form a group.

        Args:
            user_location (UserLocation): User's location data.

        Returns:
            Optional[ClusterGroup]: The group the user was added to, or None.
        """
        if user_location.user_id in self.users:  # todo: ممکنه دونفر که قبلا همگروهی داشتن، همگروهی هاشون حذف بشن و اینا بخوان با هم همگروه بشن.
            return self.users[user_location.user_id].group

        orig_user_buckets, org_node = self._get_nearest_nodes(user_location.origin_coords)
        dest_user_buckets, dest_node = self._get_nearest_nodes(user_location.destination_coords)

        candidate_users = self._collect_candidate_users(orig_user_buckets, dest_user_buckets)

        user = self._create_user(
            user_location,
            orig_user_buckets,
            dest_user_buckets,
            org_node,
            dest_node,
        )
        group = self._form_group(user, candidate_users)
        if not group:
            self._add_user_to_buckets(user)

        return group

    def remove_user(self, user_location: UserLocation) -> Optional[ClusterGroup]:
        """
        Remove a user from the clustering engine and their group.

        Args:
            user_location (UserLocation): User's location data.

        Returns:
            Optional[ClusterGroup]: The group the user was removed from, or None.
        """
        user_id = user_location.user_id
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        group = user.group

        # Remove user from buckets (if exists as individual)
        for b in user.orig_buckets:
            if b in self.orig_buckets and user_id in self.orig_buckets[b]:
                del self.orig_buckets[b][user_id]
        for b in user.dest_buckets:
            if b in self.dest_buckets and user_id in self.dest_buckets[b]:
                del self.dest_buckets[b][user_id]

        # If user is not in a group, simply remove them
        if group is None:
            del self.users[user_id]
            return None

        # Remove user from group
        left_user_id = group.update(user.user_location, remove=True)

        # If group is expired (only one user left)
        if group.status == "expired":
            left_user = self.users.get(left_user_id)
            if left_user:
                # Remove group from all buckets
                for b in group.orig_buckets.keys():
                    if group.group_id in self.orig_buckets[b]:
                        del self.orig_buckets[b][group.group_id]
                for b in group.dest_buckets.keys():
                    if group.group_id in self.dest_buckets[b]:
                        del self.dest_buckets[b][group.group_id]
                
                # Add left user as individual
                self._add_user_to_buckets(left_user)
                left_user.update_group(None)
            
            # Remove the expired user
            del self.users[user_id]
            return group.to_cluster_group()

        # If group still exists (has 2 users)
        else:
            # Update group buckets
            self._update_group_buckets(group)
            del self.users[user_id]
            return group.to_cluster_group()

    def cluster_users(self, user_locations: List[UserLocation]) -> List[ClusterGroup]:
        """
        Cluster multiple users into groups based on their locations.

        Args:
            user_locations (List[UserLocation]): List of user locations to cluster.

        Returns:
            List[ClusterGroup]: List of formed groups.
        """
        try:
            with AdvancedResourceMonitor() as monitor:
                print(f"[Engine] Clustering {len(user_locations)} user(s)…")
                internal_groups: Set[InternalClusterGroup] = set()
                for user_location in user_locations:
                    group = self.add_user(user_location)
                    if group:
                        internal_groups.add(group)
                
                return [group.to_cluster_group() for group in internal_groups]
        except Exception as e:
            print("Complete traceback:")
            traceback.print_exc()
            print(f"Error code: {e.args[0] if e.args else 'No error code'}")

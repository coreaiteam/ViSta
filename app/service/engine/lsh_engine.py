from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass, field
import numpy as np
import osmnx as ox
import networkx as nx
from sklearn.neighbors import BallTree
import pickle
import random
import uuid
import math
from ..models import UserLocation, ClusterGroup


# Define the geographical area for the street network graph
place = "Savojbolagh County, Alborz Province, Iran"
# Load the walkable street network graph from OpenStreetMap
G = ox.graph_from_place(place, network_type='walk')


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
    status: str = "forming"  # Group status, either 'forming' or 'complete'

    def __eq__(self, other):
        """Compare two InternalClusterGroup instances for equality based on group_id."""
        if not isinstance(other, InternalClusterGroup):
            return False
        return self.group_id == other.group_id

    def __hash__(self):
        """Generate hash based on group_id."""
        return hash(self.group_id)

    def _calculate_meeting_points(
        self, users: List[UserLocation]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """
        Calculate the average meeting points for origin and destination coordinates.

        Args:
            users (List[UserLocation]): List of user locations in the group.

        Returns:
            Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]: 
            Origin and destination meeting point coordinates, or None if no users.
        """
        if not users:
            return None, None
        origin_lats = [u.origin_lat for u in users]
        origin_lngs = [u.origin_lng for u in users]
        origin_meeting = (np.mean(origin_lats), np.mean(
            origin_lngs))  # Average origin coordinates
        dest_lats = [u.destination_lat for u in users]
        dest_lngs = [u.destination_lng for u in users]
        # Average destination coordinates
        dest_meeting = (np.mean(dest_lats), np.mean(dest_lngs))
        return origin_meeting, dest_meeting

    def update(self, new_user_location: UserLocation, remove=False):
        """
        Update the group by adding or removing a user and recalculate meeting points.

        Args:
            new_user_location (UserLocation): User location to add or remove.
            remove (bool): If True, remove the user; otherwise, add them.

        Returns:
            Optional[int]: ID of the user left alone if group expires, else None.
        """
        if remove:
            self.users.remove(new_user_location)
            if len(self.users) == 1:
                self.status = "expired"  # Mark group as expired if only one user remains
                return self.users[0].user_id
        else:
            self.users.append(new_user_location)  # Add new user to the group
        # Recalculate meeting points after updating users
        self.meeting_point_origin, self.meeting_point_destination = self._calculate_meeting_points(
            self.users)
        # Update status based on group size
        self.status = "complete" if len(self.users) == 3 else "forming"
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
    group: InternalClusterGroup  # Group the user belongs to, if any
    buckets: List[int]  # LSH buckets for clustering
    signature: List[int]  # MinHash signature for the user
    orig_node: int  # nearest node to origin location
    dest_node: int  # nearest node to destination location

    @property
    def origin_coords(self) -> Tuple[float, float]:
        """Get the user's origin coordinates."""
        return self.user_location.origin_coords

    @property
    def destination_coords(self) -> Tuple[float, float]:
        """Get the user's destination coordinates."""
        return self.user_location.destination_coords

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

    def __init__(self, place: str = "Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran",
                 k_nearest: int = 100, similarity_threshold: float = 0.5,
                 cache_file: str = "signatures.pkl"):
        """
        Initialize the clustering engine with geographical and clustering parameters.

        Args:
            place (str): Geographical area for the street network.
            k_nearest (int): Number of nearest nodes to consider for signatures.
            similarity_threshold (float): Minimum similarity for clustering users.
            cache_file (str): File to store precomputed signature data.
        """
        self.place = place
        self.k_nearest = k_nearest
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.G = None  # Street network graph
        self.nodes_list = None  # List of graph nodes
        self.node_to_idx = None  # Mapping of nodes to indices
        # Cache for precomputed nearest nodes
        self.nearest_nodes_cache: Dict[int, Dict[int, float]] = {}
        self.signature_cache = {}  # Cache for precomputed signatures
        self.ball_tree = None  # BallTree for nearest neighbor search
        self.node_coords = None  # Coordinates of graph nodes
        self._init_min_hashing()  # Initialize MinHash parameters
        self.buckets: Dict[int, List[int]] = {}  # LSH buckets for users
        self.users: Dict[int, User] = {}  # Dictionary of users
        self._load_or_compute_graph()  # Load street network graph
        self._build_ball_tree()  # Build BallTree for spatial queries
        self._load_or_compute_precomputed_data()  # Load or compute signatures

    def _init_min_hashing(self):
        """Initialize parameters for MinHash and Locality Sensitive Hashing (LSH)."""
        self.SNN = 100000000  # Large number for hashing
        self.r = 2  # Number of rows per band
        self.b = 15  # Number of bands
        self.M = self.r * self.b  # Total number of hash functions
        random.seed(42)  # Set random seed for reproducibility
        self.random_a = [random.randint(100, 400) for i in range(
            self.M)]  # Random coefficients for hashing
        self.random_b = [random.randint(1, 200) for i in range(
            self.M)]  # Random offsets for hashing
        self.p = 34359738421  # Large prime for hashing

    def _load_or_compute_graph(self):
        """Load the street network graph from OpenStreetMap."""
        print(f"Loading graph for {self.place}...")
        self.G = ox.graph_from_place(self.place, network_type='walk')
        self.nodes_list = list(self.G.nodes())  # Get list of graph nodes
        self.node_to_idx = {node: idx for idx, node in enumerate(
            self.nodes_list)}  # Map nodes to indices
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
        """Load precomputed signatures from cache or compute them if not available."""
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.signature_cache = cache_data['signature']
                self.nearest_nodes_cache = cache_data['nearest_nodes']
                print("Loaded precomputed data from cache")
        except FileNotFoundError:
            print("Cache not found. Computing precomputed data...")
            self._precompute_data()
            self._save_cache()

    def _my_hash(self, x: int, i: int, N: int):
        """
        Compute a hash value for MinHash.

        Args:
            x (int): Input value to hash.
            i (int): Index of hash function.
            N (int): Modulus for hash function.

        Returns:
            int: Hashed value.
        """
        return ((self.random_a[i] * x + self.random_b[i]) % self.p) % N

    def _signature(self, nearest_nodes: List[int]):
        """
        Compute MinHash signature for a set of nodes.

        Args:
            nearest_nodes (List[int]): List of nearest node indices.

        Returns:
            List[int]: MinHash signature.
        """
        sig = [self.SNN + 1 for _ in range(self.M)]  # Initialize signature
        for j in nearest_nodes:
            for i in range(self.M):
                k = self._my_hash(j, i, self.SNN)
                if k < sig[i]:
                    sig[i] = k  # Update signature with minimum hash
        return sig

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

    def _precompute_data(self):
        """Precompute signatures for all nodes in the graph."""
        print("Precomputing nearest nodes and distances...")
        for i, node in enumerate(self.nodes_list):
            if i % 100 == 0:
                print(f"Processing node {i}/{len(self.nodes_list)}")
            try:
                # Compute shortest path distances using Dijkstra's algorithm
                distances = nx.single_source_dijkstra_path_length(
                    self.G, node, cutoff=5000, weight='length'
                )
                sorted_distances = sorted(
                    distances.items(), key=lambda x: x[1])[:200]

                # Store nearest nodes
                nearest_nodes = {target_node: dist
                                 for target_node, dist in sorted_distances}
                self.nearest_nodes_cache[node] = nearest_nodes
                self.signature_cache[node] = self._signature(
                    [target_node for target_node, _ in sorted_distances])
            except Exception as e:
                print(f"Error processing node {node}: {e}")
                # Fallback to Euclidean distances if Dijkstra fails
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

                euclidean_distances.sort(key=lambda x: x[1])
                self.nearest_nodes_cache[node] = euclidean_distances[:200]

                self.signature_cache[node] = self._signature(
                    [target_node for target_node, _ in euclidean_distances[:200]])

    def _save_cache(self):
        """Save precomputed signatures to a cache file."""
        cache_data = {
            'signature': self.signature_cache,
            'nearest_nodes': self.nearest_nodes_cache,
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Precomputed data saved to cache")

    def _get_signature(self, coords: Tuple[float, float]) -> List[int]:
        """
        Get the MinHash signature for a given coordinate.

        Args:
            coords (Tuple[float, float]): Latitude and longitude coordinates.

        Returns:
            List[int]: MinHash signature of the nearest node.
        """
        lat, lng = coords
        query_point = np.array([[np.radians(lat), np.radians(lng)]])
        _, indices = self.ball_tree.query(
            query_point, k=1)  # Find nearest node
        nearest_node_idx = indices[0][0]
        nearest_node = self.nodes_list[nearest_node_idx]
        return self.signature_cache[nearest_node], nearest_node

    def _merge_signature(self, origin_sig: List[int], dest_sig: List[int], x: int) -> List[int]:
        """
        Merge origin and destination signatures.

        Args:
            origin_sig (List[int]): Origin MinHash signature.
            dest_sig (List[int]): Destination MinHash signature.
            x (int): Number of elements to take from origin signature per iteration.

        Returns:
            List[int]: Merged signature.
        """
        merged = []
        i = j = 0
        len1, len2 = len(origin_sig), len(dest_sig)
        while i < len1 or j < len2:
            for _ in range(x):
                if i < len1:
                    merged.append(origin_sig[i])
                    i += 1
            if j < len2:
                merged.append(dest_sig[j])
                j += 1
        return merged

    def _bands_hashing(self, x: List[int]) -> int:
        """
        Compute hash for a band in LSH.

        Args:
            x (List[int]): Signature segment for the band.

        Returns:
            int: Hashed value for the band.
        """
        out = 0
        for i in range(int(self.r)):
            out += x[i]
        return out % self.SNN

    def _lsh(self, sig: List[int]) -> List[int]:
        """
        Perform Locality Sensitive Hashing on a signature.

        Args:
            sig (List[int]): MinHash signature.

        Returns:
            List[int]: List of bucket indices.
        """
        return [self._bands_hashing(sig[i * self.r: (i + 1) * self.r]) for i in range(self.b)]

    def _similarity(self, user1: User, user2: User) -> float:
        orig_nearest_nodes = self.nearest_nodes_cache[user2.orig_node]
        dest_nearest_nodes = self.nearest_nodes_cache[user2.dest_node]

        orig_dist = orig_nearest_nodes.get(user1.orig_node, None)
        dest_dist = dest_nearest_nodes.get(user1.dest_node, None)

        if orig_dist is None or dest_dist is None:
            return -1.0

        total_distance = orig_dist + dest_dist

        if total_distance == 0:
            return 1.0

        similarity = math.exp(-total_distance / 1000)

        return similarity

    def _collect_candidate_users(self, user_buckets: List[int]) -> set[int]:
        """
        Collect candidate users from LSH buckets.

        Args:
            user_signature (List[int]): User's MinHash signature.
            user_buckets (List[int]): User's LSH bucket indices.

        Returns:
            set[int]: Set of candidate user IDs.
        """
        users = set()
        for b in user_buckets:
            users.update(self.buckets.get(b, []))  # Add users from each bucket
        return users

    def _update_or_create_group(self, group_users: List[User]) -> InternalClusterGroup:
        """
        Update an existing group or create a new one with the given users.

        Args:
            group_users (List[User]): List of users to form or update the group.

        Returns:
            InternalClusterGroup: The updated or newly created group.
        """
        if len(group_users) > 2:
            if group_users[2].companions_number == 1 and group_users[1].companions_number == 0:
                group = group_users[2].group
            elif group_users[1].companions_number == 1:
                group = group_users[1].group
            else:
                group = InternalClusterGroup(
                    users=[u.user_location for u in group_users[1:]],
                    created_at=datetime.now(timezone.utc),
                )
                group_users[1].update_group(group)
                group_users[2].update_group(group)
        elif group_users[1].companions_number == 1:
            group = group_users[1].group
        else:
            group = InternalClusterGroup(
                users=[group_users[1].user_location],
                created_at=datetime.now(timezone.utc),
            )
            group_users[1].update_group(group)
        # Add the first user to the group
        group.update(group_users[0].user_location)
        # Update the user's group assignment
        group_users[0].update_group(group)

        return group

    def _form_group(self, user: User, candidate_users: set[int]) -> InternalClusterGroup:
        """
        Form a group for a user by finding similar users from candidates.

        Args:
            user (User): The user to cluster.
            candidate_users (set[int]): Set of candidate user IDs.

        Returns:
            Optional[InternalClusterGroup]: The formed group, or None if no group formed.
        """
        near_users: List[Tuple[User, int]] = []

        for candid_id in candidate_users:
            candid = self.users[candid_id]

            # Calculate similarity based on Euclidean distance
            sim = self._similarity(user, candid)

            if sim >= self.similarity_threshold:
                if candid.companions_number < 2:
                    near_users.append((candid, sim))
        if len(near_users) == 0:
            return None
        if len(near_users) == 1:
            return self._update_or_create_group([user, near_users[0][0]])

        near_users.sort(key=lambda x: x[1], reverse=True)
        return self._update_or_create_group([user, near_users[0][0], near_users[1][0]])

    def _create_user(self, user_location: UserLocation, buckets: List[int],
                     signature: List[int], orig_node: int, dest_node: int) -> User:
        """
        Create a new User instance.

        Args:
            user_location (UserLocation): User's location data.
            buckets (List[int]): LSH buckets for the user.
            signature (List[int]): MinHash signature.

        Returns:
            User: The created user instance.
        """

        new_user = User(
            user_id=user_location.user_id,
            user_location=user_location,
            group=None,
            buckets=buckets,
            signature=signature,
            orig_node=orig_node,
            dest_node=dest_node,
        )
        self.users[new_user.user_id] = new_user
        return new_user

    def _add_user_to_buckets(self, user: User):
        """Add a user to their corresponding LSH buckets."""
        for b in user.buckets:
            if b in self.buckets.keys():
                self.buckets[b].append(user.user_id)
            else:
                self.buckets[b] = [user.user_id]

    def add_user(self, user_location: UserLocation) -> InternalClusterGroup:
        """
        Add a user to the clustering engine and attempt to form a group.

        Args:
            user_location (UserLocation): User's location data.

        Returns:
            Optional[ClusterGroup]: The group the user was added to, or None.
        """
        if user_location.user_id in self.users.keys():
            return self.users[user_location.user_id].group

        origin_sig, org_node = self._get_signature(user_location.origin_coords)
        dest_sig, dest_node = self._get_signature(
            user_location.destination_coords)
        user_signature = self._merge_signature(origin_sig, dest_sig, 1)

        user_buckets = self._lsh(user_signature)
        candidate_users = self._collect_candidate_users(user_buckets)

        user = self._create_user(
            user_location,
            user_buckets,
            user_signature,
            org_node,
            dest_node,
        )
        group = self._form_group(user, candidate_users)
        self._add_user_to_buckets(user)

        return group if group else None

    def remove_user(self, user_location: UserLocation) -> Optional[ClusterGroup]:
        """
        Remove a user from the clustering engine and their group.

        Args:
            user_location (UserLocation): User's location data.

        Returns:
            Optional[ClusterGroup]: The group the user was removed from, or None.
        """
        group = None
        if user_location.user_id in self.users.keys():
            user = self.users.pop(user_location.user_id)
            for b in user.buckets:
                # Remove user from buckets
                self.buckets[b].remove(user.user_id)
            if user.companions_number > 0:
                group = user.group
                alone_user_id = group.update(user_location, remove=True)
                if alone_user_id:
                    self.users[alone_user_id].update_group(
                        group)  # Update group for remaining user
        return group.to_cluster_group() if group else None

    def remove_group_users(self, group: ClusterGroup):
        """
        Remove all users from a specified group and mark it as expired.

        Args:
            group (ClusterGroup): The group to dissolve.
        """
        internal_group = InternalClusterGroup(
            group_id=group.group_id,
            users=group.users,
            created_at=group.created_at,
            meeting_point_origin=group.meeting_point_origin,
            meeting_point_destination=group.meeting_point_destination,
            status=group.status
        )
        if internal_group.status == "expired":
            return
        internal_group.status = "expired"  # Mark group as expired
        for user_location in internal_group.users:
            if user_location.user_id in self.users.keys():
                user = self.users.pop(user_location.user_id)
                for b in user.buckets:
                    # Remove user from buckets
                    self.buckets[b].remove(user.user_id)

    def cluster_users(self, user_locations: List[UserLocation]) -> List[ClusterGroup]:
        """
        Cluster multiple users into groups based on their locations.

        Args:
            user_locations (List[UserLocation]): List of user locations to cluster.

        Returns:
            List[ClusterGroup]: List of formed groups.
        """
        print(f"[Engine] Clustering {len(user_locations)} user(s)…")
        internal_groups: set[InternalClusterGroup] = set()
        for user_location in user_locations:
            group = self.add_user(user_location)
            if group:
                internal_groups.add(group)
        return [group.to_cluster_group() for group in internal_groups]

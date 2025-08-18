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
from .models import UserLocation, ClusterGroup

# Define the geographical area for the graph
place = "Savojbolagh County, Alborz Province, Iran"
# Load the walkable street network graph from OpenStreetMap
G = ox.graph_from_place(place, network_type='walk')


@dataclass
class InternalClusterGroup:
    group_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    users: List[UserLocation] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    meeting_point_origin: Optional[Tuple[float, float]] = None
    meeting_point_destination: Optional[Tuple[float, float]] = None
    status: str = "forming"

    def __eq__(self, other):
        if not isinstance(other, InternalClusterGroup):
            return False
        return self.group_id == other.group_id

    def __hash__(self):
        return hash(self.group_id)

    def _calculate_meeting_points(
        self, users: List[UserLocation]
    ) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        if not users:
            return None, None
        origin_lats = [u.origin_lat for u in users]
        origin_lngs = [u.origin_lng for u in users]
        origin_meeting = (np.mean(origin_lats), np.mean(origin_lngs))
        dest_lats = [u.destination_lat for u in users]
        dest_lngs = [u.destination_lng for u in users]
        dest_meeting = (np.mean(dest_lats), np.mean(dest_lngs))
        return origin_meeting, dest_meeting

    def update(self, new_user_location: UserLocation, remove=False):
        if remove:
            self.users.remove(new_user_location)
            if len(self.users) == 1:
                self.status = "expired"
                return self.users[0].user_id
        else:
            self.users.append(new_user_location)
        self.meeting_point_origin, self.meeting_point_destination = self._calculate_meeting_points(self.users)
        self.status = "complete" if len(self.users) == 3 else "forming"
        return None

    def to_cluster_group(self) -> ClusterGroup:
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
    user_id: int
    user_location: UserLocation
    group: Optional[InternalClusterGroup]
    buckets: List[int]
    signature: List[int]

    @property
    def origin_coords(self) -> Tuple[float, float]:
        return self.user_location.origin_coords

    @property
    def destination_coords(self) -> Tuple[float, float]:
        return self.user_location.destination_coords

    @property
    def companions_number(self) -> int:
        if self.group:
            return len(self.group.users) - 1
        return 0

    def update_group(self, group: Optional[InternalClusterGroup]):
        self.group = group

    def to_dict(self) -> Dict:
        return {}

class ClusteringEngine:
    def __init__(self, place: str = "Savojbolagh Central District, Savojbolagh County, Alborz Province, Iran", 
                 k_nearest: int = 100, similarity_threshold: float = 0.7,
                 cache_file: str = "signatures.pkl"):
        self.place = place
        self.k_nearest = k_nearest
        self.similarity_threshold = similarity_threshold
        self.cache_file = cache_file
        self.G = None
        self.nodes_list = None
        self.node_to_idx = None
        self.signature_cache = {}
        self.ball_tree = None
        self.node_coords = None
        self._init_min_hashing()
        self.buckets: Dict[int, List[int]] = {}
        self.users: Dict[int, User] = {}
        self._load_or_compute_graph()
        self._build_ball_tree()
        self._load_or_compute_precomputed_data()

    def _init_min_hashing(self):
        self.SNN = 100000000
        self.r = 2
        self.b = 15
        self.M = self.r * self.b
        random.seed(42)
        self.random_a = [random.randint(100, 400) for i in range(self.M)]
        self.random_b = [random.randint(1, 200) for i in range(self.M)]
        self.p = 34359738421

    def _load_or_compute_graph(self):
        print(f"Loading graph for {self.place}...")
        self.G = ox.graph_from_place(self.place, network_type='walk')
        self.nodes_list = list(self.G.nodes())
        self.node_to_idx = {node: idx for idx, node in enumerate(self.nodes_list)}
        print(f"Graph loaded with {len(self.nodes_list)} nodes")

    def _build_ball_tree(self):
        print("Building BallTree for fast nearest neighbor search...")
        self.node_coords = []
        for node in self.nodes_list:
            lat = np.radians(self.G.nodes[node]['y'])
            lng = np.radians(self.G.nodes[node]['x'])
            self.node_coords.append([lat, lng])
        self.node_coords = np.array(self.node_coords)
        self.ball_tree = BallTree(self.node_coords, metric='haversine')
        print("BallTree built successfully")

    def _load_or_compute_precomputed_data(self):
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                self.signature_cache = cache_data['signature']
                print("Loaded precomputed data from cache")
        except FileNotFoundError:
            print("Cache not found. Computing precomputed data...")
            self._precompute_data()
            self._save_cache()

    def _my_hash(self, x: int, i: int, N: int):
        return ((self.random_a[i] * x + self.random_b[i]) % self.p) % N

    def _signature(self, nearest_nodes: List[int]):
        sig = [self.SNN + 1 for _ in range(self.M)]
        for j in nearest_nodes:
            for i in range(self.M):
                k = self._my_hash(j, i, self.SNN)
                if k < sig[i]:
                    sig[i] = k
        return sig

    def _precompute_data(self):
        print("Precomputing nearest nodes and distances...")
        for i, node in enumerate(self.nodes_list):
            if i % 100 == 0:
                print(f"Processing node {i}/{len(self.nodes_list)}")
            try:
                distances = nx.single_source_dijkstra_path_length(
                    self.G, node, cutoff=5000, weight='length'
                )
                sorted_distances = sorted(distances.items(), key=lambda x: x[1])[:200]
                self.signature_cache[node] = self._signature([target_node for target_node, _ in sorted_distances])
            except Exception as e:
                print(f"Error processing node {node}: {e}")
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
                self.signature_cache[node] = self._signature([target_node for target_node, _ in euclidean_distances[:200]])

    def _save_cache(self):
        cache_data = {
            'signature': self.signature_cache,
        }
        with open(self.cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print("Precomputed data saved to cache")

    def _get_signature(self, coords: Tuple[float, float]) -> List[int]:
        lat, lng = coords
        query_point = np.array([[np.radians(lat), np.radians(lng)]])
        _, indices = self.ball_tree.query(query_point, k=1)
        nearest_node_idx = indices[0][0]
        nearest_node = self.nodes_list[nearest_node_idx]
        return self.signature_cache[nearest_node]

    def _merge_signature(self, origin_sig: List[int], dist_sig: List[int], x: int) -> List[int]:
        merged = []
        i = j = 0
        len1, len2 = len(origin_sig), len(dist_sig)
        while i < len1 or j < len2:
            for _ in range(x):
                if i < len1:
                    merged.append(origin_sig[i])
                    i += 1
            if j < len2:
                merged.append(dist_sig[j])
                j += 1
        return merged

    def _bands_hashing(self, x: List[int]) -> int:
        out = 0
        for i in range(int(self.r)):
            out += x[i]
        return out % self.SNN

    def _lsh(self, sig: List[int]) -> List[int]:
        return [self._bands_hashing(sig[i * self.r: (i + 1) * self.r]) for i in range(self.b)]

    def _similarity(self, sig1: List[int], sig2: List[int]) -> float:
        if len(sig1) != len(sig2):
            raise ValueError("Lengths of the lists must be equal")
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def _collect_candidate_users(self, user_signature: List[int], user_buckets: List[int]) -> set[int]:
        users = set()
        for b in user_buckets:
            users.update(self.buckets.get(b, []))
        return users

    def _update_or_create_group(self, group_users: List[User]) -> InternalClusterGroup:
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
        elif group_users[1].companions_number == 1:
            group = group_users[1].group
        else:
            group = InternalClusterGroup(
                users=[group_users[1].user_location],
                created_at=datetime.now(timezone.utc),
            )
        group.update(group_users[0].user_location)
        group_users[0].update_group(group)
        return group

    def _form_group(self, user: User, candidate_users: set[int]) -> Optional[InternalClusterGroup]:
        u_sig: List[int] = user.signature
        near_users: List[Tuple[User, int]] = []
        for candid_id in candidate_users:
            candid = self.users[candid_id]
            sim = self._similarity(u_sig, candid.signature)
            if sim >= self.similarity_threshold:
                if candid.companions_number < 2:
                    near_users.append((candid, sim))
        if len(near_users) == 0:
            return None
        if len(near_users) == 1:
            return self._update_or_create_group([user, near_users[0][0]])
        near_users.sort(key=lambda x: x[1], reverse=True)
        return self._update_or_create_group([user, near_users[0][0], near_users[1][0]])

    def _create_user(self, user_location: UserLocation, buckets: List[int], signature: List[int]) -> User:
        new_user = User(
            user_id=user_location.user_id,
            user_location=user_location,
            group=None,
            buckets=buckets,
            signature=signature
        )
        self.users[new_user.user_id] = new_user
        return new_user

    def _add_user_to_buckets(self, user: User):
        for b in user.buckets:
            if b in self.buckets.keys():
                self.buckets[b].append(user.user_id)
            else:
                self.buckets[b] = [user.user_id]

    def add_user(self, user_location: UserLocation) -> Optional[ClusterGroup]:
        if user_location.user_id in self.users.keys():
            group = self.users[user_location.user_id].group
            return group.to_cluster_group() if group else None
        origin_sig = self._get_signature(user_location.origin_coords)
        dist_sig = self._get_signature(user_location.destination_coords)
        user_signature = self._merge_signature(origin_sig, dist_sig, 1)
        user_buckets = self._lsh(user_signature)
        candidate_users = self._collect_candidate_users(user_signature, user_buckets)
        user = self._create_user(user_location, user_buckets, user_signature)
        group = self._form_group(user, candidate_users)
        self._add_user_to_buckets(user)
        return group.to_cluster_group() if group else None

    def remove_user(self, user_location: UserLocation) -> Optional[ClusterGroup]:
        group = None
        if user_location.user_id in self.users.keys():
            user = self.users.pop(user_location.user_id)
            for b in user.buckets:
                self.buckets[b].remove(user.user_id)
            if user.companions_number > 0:
                group = user.group
                alone_user_id = group.update(user_location, remove=True)
                if alone_user_id:
                    self.users[alone_user_id].update_group(group)
        return group.to_cluster_group() if group else None

    def remove_group_users(self, group: ClusterGroup):
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
        internal_group.status = "expired"
        for user_location in internal_group.users:
            if user_location.user_id in self.users.keys():
                user = self.users.pop(user_location.user_id)
                for b in user.buckets:
                    self.buckets[b].remove(user.user_id)

    def cluster_users(self, user_locations: List[UserLocation]) -> List[ClusterGroup]:
        print(f"[Engine] Clustering {len(user_locations)} user(s)…")
        internal_groups = set()
        for user_location in user_locations:
            group = self.add_user(user_location)
            if group:
                internal_groups.add(self.users[user_location.user_id].group)
        return [group.to_cluster_group() for group in internal_groups]
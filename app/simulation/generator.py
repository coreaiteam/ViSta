import os
import json
import random
import networkx as nx
from typing import List, Tuple, Optional
from sklearn.neighbors import BallTree
import numpy as np


# =========================
#   Data Generator Class
# =========================
class OSMDataGenerator:
    """
    Generate synthetic test data (origin-destination pairs) 
    from an OSM graph using network distances (not Euclidean).
    """

    def __init__(
        self,
        graph: nx.MultiDiGraph,
        num_main_points: int = 2,
        neighbors_k: int = 100,
        sample_size: int = 20,
        max_walk_dist: int = 500,
        seed: int = 42,
        use_predefined_points: bool = False,  # New parameter
        predefined_data: List[dict] = None   # New parameter
    ):
        self.graph = graph
        self.use_predefined_points = use_predefined_points
        self.predefined_data = predefined_data if predefined_data else []
        self.num_main_points = num_main_points
        self.neighbors_k = neighbors_k
        self.sample_size = sample_size
        self.max_walk_dist = max_walk_dist
        self.random = random.Random(seed)

        # placeholders for BallTree
        self.balltree = None
        self.node_ids = None
        self.coords_rad = None

        self._build_balltree()

    def _build_balltree(self):
        """
        Build a BallTree from the graph nodes for fast nearest neighbor queries.
        Call this once after graph is loaded.
        """
        self.node_ids = list(self.graph.nodes)
        coords = np.array(
            [(self.graph.nodes[n]["y"], self.graph.nodes[n]["x"])
             for n in self.node_ids]
        )

        # Convert to radians for haversine metric
        self.coords_rad = np.radians(coords)
        self.balltree = BallTree(self.coords_rad, metric="haversine")

    def _get_random_node(self) -> int:
        """
        Pick a random node from the graph.
        """
        return self.random.choice(self.node_ids)

    def _get_main_node(self, node_type: str = "origin", i: int = 0) -> int:
        """
        Pick a node from predefined data or random node.
        """
        if self.use_predefined_points and i < len(self.predefined_data):
            entry = self.predefined_data[i]
            if node_type == "origin":
                target_lat, target_lon = entry["origin"]
            else:
                target_lat, target_lon = entry["dest"]
            
            closest_node = self._find_closest_node(target_lat, target_lon)
            return closest_node
        else:
            # Generate random node if no predefined data available
            return self._get_random_node()

    def _find_closest_node(self, target_lat: float, target_lon: float) -> int:
        """
        Find the closest graph node using BallTree (Haversine distance).
        """
        if self.balltree is None:
            raise RuntimeError(
                "BallTree not built. Call _build_balltree() first.")

        target_rad = np.radians([[target_lat, target_lon]])
        dist, ind = self.balltree.query(target_rad, k=1)

        # Index lookup
        node_id = self.node_ids[ind[0][0]]
        return node_id

    def _get_nearby_nodes(self, node: int) -> List[int]:
        """
        Find nearby nodes based on walking distance (network distance).
        """
        lengths = nx.single_source_dijkstra_path_length(
            self.graph, node, cutoff=self.max_walk_dist, weight="length"
        )
        nearby_nodes = list(lengths.keys())
        return nearby_nodes

    def _node_to_latlon(self, node: int) -> Tuple[float, float]:
        """Convert node ID to (lat, lon)."""
        return self.graph.nodes[node]["y"], self.graph.nodes[node]["x"]

    def generate(self) -> List[Tuple[float, float, float, float]]:
        """
        Generate dataset of (lat_o, lon_o, lat_d, lon_d).
        """
        data = []
        used_origins = set()
        used_dests = set()
        cnt = 0

        for i in range(self.num_main_points):
            origin = self._get_main_node('origin', i)
            dest = self._get_main_node('dest', i)

            nearby_origins = self._get_nearby_nodes(origin)
            nearby_dests = self._get_nearby_nodes(dest)

            available_origins = [
                n for n in nearby_origins if n not in used_origins]
            available_dests = [n for n in nearby_dests if n not in used_dests]

            if len(available_origins) > self.neighbors_k:
                available_origins = self.random.sample(
                    available_origins, self.neighbors_k)
            if len(available_dests) > self.neighbors_k:
                available_dests = self.random.sample(
                    available_dests, self.neighbors_k)

            self.random.shuffle(available_origins)
            self.random.shuffle(available_dests)

            num_pairs = min(self.sample_size, len(
                available_origins), len(available_dests))
            for j in range(num_pairs):
                o = available_origins[j]
                d = available_dests[j]
                used_origins.add(o)
                used_dests.add(d)
                lat_o, lon_o = self._node_to_latlon(o)
                lat_d, lon_d = self._node_to_latlon(d)
                data.append((lat_o, lon_o, lat_d, lon_d))
            
            cnt += num_pairs
            if cnt % 100 == 0:
                print(f"{cnt}/{num_pairs*self.num_main_points} data generated")

        return data


# =========================
#   Data Storage Class
# =========================
class DataStorage:
    """
    Handle saving and loading dataset to/from disk.
    """

    def __init__(self, filepath: str = "osm_data.json"):
        self.filepath = filepath

    def save(self, data: List[Tuple[float, float, float, float]]) -> None:
        """Save dataset to JSON file."""
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def load(self) -> Optional[List[Tuple[float, float, float, float]]]:
        """Load dataset if exists."""
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", encoding="utf-8") as f:
                return json.load(f)
        return None


# =========================
#   Dataset Class
# =========================
class OSMDataset:
    """
    Dataset wrapper: can be used in iterator or batch mode.
    """

    def __init__(self, data: List[Tuple[float, float, float, float]]):
        self.data = data

    def __iter__(self):
        """Stream data sample by sample."""
        for sample in self.data:
            yield sample

    def as_batch(self) -> List[Tuple[float, float, float, float]]:
        """Return all data at once."""
        return self.data


# =========================
#   High-level API
# =========================
class OSMTestDataPipeline:
    """
    High-level API combining Generator, Storage, and Dataset.
    """

    def __init__(self, graph, storage_path="osm_data.json", use_predefined_points=False, predefined_data=None, **gen_params):
        self.generator = OSMDataGenerator(
            graph, 
            use_predefined_points=use_predefined_points,
            predefined_data=predefined_data,
            **gen_params
        )
        self.storage = DataStorage(storage_path)
        self.dataset: Optional[OSMDataset] = None

    def prepare(self, force_recompute: bool = False) -> None:
        """
        Prepare dataset: load if exists, otherwise generate & save.
        """
        if not force_recompute:
            data = self.storage.load()
            if data is not None:
                self.dataset = OSMDataset(data)
                return

        # generate fresh data
        data = self.generator.generate()
        self.storage.save(data)
        self.dataset = OSMDataset(data)

    def get_dataset(self) -> OSMDataset:
        """Return dataset object."""
        if self.dataset is None:
            raise RuntimeError("Dataset not prepared. Call prepare() first.")
        return self.dataset

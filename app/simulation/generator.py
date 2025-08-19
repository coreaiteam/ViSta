import os
import json
import random
import networkx as nx
from typing import List, Tuple, Optional


data = [
    {
        'origin': [35.97897, 50.73145],
        'dest': [35.962546, 50.678285]
    },
    {
        'origin': [35.98123, 50.74567],
        'dest': [35.98368, 50.74327]
    },
    {
        'origin': [35.962468, 50.688440],
        'dest': [35.958855, 50.672861]
    }
]


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
        num_main_points: int = 10,
        neighbors_k: int = 100,
        sample_size: int = 20,
        max_walk_dist: int = 500,
        seed: int = 42,
    ):
        self.graph = graph
        self.lookup_data = data
        self.num_main_points = num_main_points
        self.neighbors_k = neighbors_k
        self.sample_size = sample_size
        self.max_walk_dist = max_walk_dist
        self.random = random.Random(seed)

    def _random_node(self, node_type: str = 'origin') -> int:
        """
        Pick a node from the lookup table based on lat/lon coordinates.

        Args:
            node_type: Either 'origin' or 'dest' to specify which coordinate to use

        Returns:
            Node ID from the graph that's closest to the lookup table coordinate
        """
        # Randomly select an entry from the lookup table
        entry = self.random.choice(self.lookup_data)

        # Get the appropriate coordinate based on node_type
        if node_type == 'origin':
            target_lat, target_lon = entry['origin']
        else:  # node_type == 'dest'
            target_lat, target_lon = entry['dest']

        # Find the closest node in the graph to this coordinate
        closest_node = self._find_closest_node(target_lat, target_lon)
        return closest_node

    def _find_closest_node(self, target_lat: float, target_lon: float) -> int:
        """
        Find the node in the graph closest to the given lat/lon coordinates.
        """
        min_distance = float('inf')
        closest_node = None

        for node_id in self.graph.nodes:
            node_lat = self.graph.nodes[node_id]["y"]
            node_lon = self.graph.nodes[node_id]["x"]

            # Calculate Euclidean distance (for finding closest node)
            distance = ((target_lat - node_lat) ** 2 +
                        (target_lon - node_lon) ** 2) ** 0.5

            if distance < min_distance:
                min_distance = distance
                closest_node = node_id

        return closest_node

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
        for _ in range(self.num_main_points):
            origin = self._random_node('origin')
            dest = self._random_node('dest')

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
            for i in range(num_pairs):
                o = available_origins[i]
                d = available_dests[i]
                used_origins.add(o)
                used_dests.add(d)
                lat_o, lon_o = self._node_to_latlon(o)
                lat_d, lon_d = self._node_to_latlon(d)
                data.append((lat_o, lon_o, lat_d, lon_d))

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

    def __init__(self, graph, storage_path="osm_data.json", **gen_params):
        self.generator = OSMDataGenerator(graph, **gen_params)
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

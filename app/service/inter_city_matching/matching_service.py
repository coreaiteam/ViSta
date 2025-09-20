from typing import List, Dict, Tuple
import threading
import time

from .models import InterCityUserLocation
from .matching_engine import InterCityMatcher, get_inter_city_matching_engine


class InterCityRideSharingSystem:
    """
    Ride sharing system that stores users, performs background matching,
    and provides access to the latest results.
    """

    def __init__(self, matching_engine: InterCityMatcher, clustering_interval: int = 5):
        """
        Initialize the ride sharing system.
        """
        self.users_by_city: Dict[str, List[InterCityUserLocation]] = {}
        self.match_results: Dict[str, List[List[InterCityUserLocation]]] = {}
        self.clustering_interval = clustering_interval
        self.matching_engine = matching_engine

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def add_user(self, user: InterCityUserLocation):
        """
        Add a new user to the system.
        """
        with self._lock:
            if user.origin_city not in self.users_by_city:
                self.users_by_city[user.origin_city] = []
            self.users_by_city[user.origin_city].append(user)

    def match_all(self) -> Dict[str, List[List[InterCityUserLocation]]]:
        """
        Run the matching algorithm for all users grouped by city.
        """
        results: Dict[str, List[List[InterCityUserLocation]]] = {}
        with self._lock:
            for city, users in self.users_by_city.items():
                results[city] = self.matching_engine.match_users_in_city(users)
        return results

    def _background_matching(self):
        """Background thread that updates matching results periodically."""
        while not self._stop_event.is_set():
            self.match_results = self.match_all()
            time.sleep(self.clustering_interval)

    def start(self):
        """
        Start the background matching service.
        """
        if self._thread and self._thread.is_alive():
            return  # already running
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._background_matching, daemon=True)
        self._thread.start()

    def stop(self):
        """
        Stop the background matching service.
        """
        if not self._thread:
            return
        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def get_results(self) -> Dict[str, List[List[InterCityUserLocation]]]:
        """
        Get the latest matching results from the background service.
        """
        return self.match_results


    def get_all_users(self) -> List[InterCityUserLocation]:
        """
        Get all users in the system as a flat list.
        """
        with self._lock:
            all_users = []
            for users in self.users_by_city.values():
                all_users.extend(users)
            return all_users

    def fetch_route_for_user(self, user: InterCityUserLocation) -> List[Tuple[float, float]]:
        """
        Fetch the route for a given user using the matching engine.
        """
        return self.matching_engine.fetch_route(user)
    


_inter_city_clustering_service: InterCityRideSharingSystem | None = None
_inter_city_matching_engine = get_inter_city_matching_engine()


def get_inter_city_clustering_service(clustering_interval=5) -> InterCityRideSharingSystem:
    global _inter_city_clustering_service

    if _inter_city_clustering_service is None:
        _inter_city_clustering_service = InterCityRideSharingSystem(
            clustering_interval=clustering_interval,
            matching_engine=_inter_city_matching_engine,
        )

    return _inter_city_clustering_service

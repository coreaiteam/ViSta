import math
import requests
from typing import List, Dict, Tuple
from datetime import timedelta
from functools import lru_cache
from sklearn.neighbors import BallTree
import numpy as np

from .models import UserLocation


def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (in km) between two points."""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


class InterCityMatcher:
    def __init__(self, max_group_size: int = 3, time_window_minutes: int = 30,
                 destination_threshold_km: int = 30, overlap_threshold_km: int = 20):
        """
        Matching engine for inter-city ride sharing.

        Args:
            max_group_size: Maximum passengers allowed in one group
            time_window_minutes: Maximum time difference between departures
            destination_threshold_km: Max distance (km) between destinations to be considered "close"
            overlap_threshold_km: Max distance (km) between route segments to consider routes overlapping
        """
        self.max_group_size = max_group_size
        self.time_window = timedelta(minutes=time_window_minutes)
        self.destination_threshold_km = destination_threshold_km
        self.overlap_threshold_km = overlap_threshold_km

        # Cache for user routes
        self._route_cache: Dict[int, List[Tuple[float, float]]] = {}

    def _fetch_route(self, user: UserLocation) -> List[Tuple[float, float]]:
        """
        Fetch and cache route polyline from OSRM for a given user.
        """
        if user.user_id in self._route_cache:
            return self._route_cache[user.user_id]

        url = (
            f"http://router.project-osrm.org/route/v1/driving/"
            f"{user.origin_lng},{user.origin_lat};{user.destination_lng},{user.destination_lat}"
            f"?overview=full&geometries=geojson"
        )
        resp = requests.get(url)
        data = resp.json()

        if "routes" not in data or len(data["routes"]) == 0:
            return []

        coords = data["routes"][0]["geometry"]["coordinates"]  # [lon, lat]
        route = [(lat, lon) for lon, lat in coords]  # convert to (lat, lon)

        # Cache result
        self._route_cache[user.user_id] = route
        return route

    def _routes_overlap(self, user1: UserLocation, user2: UserLocation,
                        similarity_threshold: float = 0.8) -> bool:
        """
        Determine if two routes overlap based on actual route similarity.
        - similarity_threshold: fraction of the shorter route that must be close to the other route
        """
        route1 = self._fetch_route(user1)
        route2 = self._fetch_route(user2)

        if not route1 or not route2:
            return False

        # تعیین مسیر کوتاه‌تر و بلندتر
        if len(route1) <= len(route2):
            short_route, long_route = route1, route2
        else:
            short_route, long_route = route2, route1

        # تبدیل به رادیان برای BallTree
        arr_long = np.radians(np.array(long_route))
        tree = BallTree(arr_long, metric="haversine")

        arr_short = np.radians(np.array(short_route))
        dist, _ = tree.query(arr_short, k=1)
        dist_km = dist.flatten() * 6371

        # تعداد نقاط نزدیک
        num_close = np.sum(dist_km <= self.overlap_threshold_km)
        fraction_close = num_close / len(short_route)

        return fraction_close >= similarity_threshold

    def _can_match(self, user1: UserLocation, user2: UserLocation, group: List[UserLocation]) -> bool:
        """Check if user2 can be added to group with user1."""

        # Origin must be the same city
        if user1.origin_city != user2.origin_city:
            return False

        # Departure time must be within allowed window
        time_diff = abs((user1.departure_time - user2.departure_time).total_seconds() / 60)
        if time_diff > self.time_window.total_seconds() / 60:
            return False

        # Check destination closeness OR route overlap
        dest_distance = haversine(
            user1.destination_lat, user1.destination_lng,
            user2.destination_lat, user2.destination_lng
        )
        if dest_distance > self.destination_threshold_km:
            if not self._routes_overlap(user1, user2):
                return False

        # Group capacity must not be exceeded
        if sum(u.passengers for u in group) + user2.passengers > self.max_group_size:
            return False

        return True

    def match_users_in_city(self, city_users: List[UserLocation]) -> List[List[UserLocation]]:
        """Match users within the same city based on origin, time, destination proximity, and route overlap."""
        matched_groups: List[List[UserLocation]] = []
        used = set()
        for i, user1 in enumerate(city_users):
            if i in used:
                continue
            group = [user1]
            used.add(i)
            for j, user2 in enumerate(city_users):
                if j in used:
                    continue
                if self._can_match(user1, user2, group):
                    group.append(user2)
                    used.add(j)
                if len(group) >= self.max_group_size:
                    break
            matched_groups.append(group)
        return matched_groups

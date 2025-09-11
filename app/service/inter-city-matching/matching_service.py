from typing import List, Dict

from .models import UserLocation
from .matching_engine import InterCityMatcher


class RideSharingSystem:
    def __init__(self):
        self.users_by_city: Dict[str, List[UserLocation]] = {}

    def add_user(self, user: UserLocation):
        if user.origin_city not in self.users_by_city:
            self.users_by_city[user.origin_city] = []
        self.users_by_city[user.origin_city].append(user)

    def match_all(self) -> Dict[str, List[List["UserLocation"]]]:
        """Run matching per city"""
        results: Dict[str, List[List["UserLocation"]]] = {}
        matcher = InterCityMatcher()
        for city, users in self.users_by_city.items():
            results[city] = matcher.match_users_in_city(users)
        return results

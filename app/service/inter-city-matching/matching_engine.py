from typing import List
from datetime import timedelta

from .models import UserLocation

class InterCityMatcher:
    def __init__(self, max_group_size: int = 3, time_window_minutes: int = 30):
        self.max_group_size = max_group_size
        self.time_window = timedelta(minutes=time_window_minutes)

    def _can_match(self, user1: UserLocation, user2: UserLocation, group: List[UserLocation]) -> bool:
        """Check if user2 can be added to group with user1"""
        if user1.origin_city != user2.origin_city:
            return False
        time_diff = abs((user1.departure_time - user2.departure_time).total_seconds() / 60)
        if time_diff > self.time_window.total_seconds() / 60:
            return False
        if sum(u.passengers for u in group) + user2.passengers > self.max_group_size:
            return False
        return True

    def match_users_in_city(self, city_users: List[UserLocation]) -> List[List[UserLocation]]:
        """Match users within the same city"""
        matched_groups: List[List["UserLocation"]] = []
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


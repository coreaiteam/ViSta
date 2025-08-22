import threading
from typing import List, Dict, Optional
from queue import Queue, Empty
from .models import UserLocation, ClusterGroup, UserStatus


class DataStorage:
    """Thread-safe storage for user locations and cluster groups"""

    def __init__(self):
        self._lock = threading.RLock()
        self.user_locations: Dict[int, UserLocation] = {}
        self.cluster_groups: Dict[str, ClusterGroup] = {}
        self.user_to_group: Dict[int, str] = {}
        self._update_queue = Queue()

    def add_user_location(self, user_location: UserLocation) -> None:
        with self._lock:
            if user_location.user_id in self.user_locations:
                return
            self.user_locations[user_location.user_id] = user_location
            self._update_queue.put(
                {"type": "user_added", "user_id": user_location.user_id}
            )

    def update_user_location(self, user_location: UserLocation) -> None:
        with self._lock:
            if user_location.user_id in self.user_locations:
                self.user_locations[user_location.user_id] = user_location
                self._update_queue.put(
                    {"type": "user_updated", "user_id": user_location.user_id})

    def remove_user_location(self, user_id: int) -> None:
        with self._lock:
            if user_id in self.user_locations:
                # Remove user from group if assigned
                if user_id in self.user_to_group:
                    self._remove_user_from_group(user_id)
                del self.user_locations[user_id]
                self._update_queue.put(
                    {"type": "user_removed", "user_id": user_id})

    def _remove_user_from_group(self, user_id: int) -> None:
        """Internal method to remove user from their group and handle group cleanup"""
        group_id = self.user_to_group[user_id]
        group = self.cluster_groups[group_id]

        # Remove user from group
        group.users = [u for u in group.users if u.user_id != user_id]
        del self.user_to_group[user_id]

        # Handle group cleanup based on remaining users
        if len(group.users) <= 1:
            # Remove group if empty or has only one user
            for user in group.users:
                user.status = UserStatus.PENDING
                if user.user_id in self.user_to_group:
                    del self.user_to_group[user.user_id]
            del self.cluster_groups[group_id]
            self._update_queue.put(
                {"type": "group_disbanded", "group_id": group_id})
        else:
            # Update group status if not complete anymore
            if not group.is_complete():
                group.status = "forming"
            self._update_queue.put(
                {"type": "group_updated", "group_id": group_id})

    def add_cluster_group(self, group: ClusterGroup) -> None:
        with self._lock:
            self.cluster_groups[group.group_id] = group
            for user in group.users:
                self.user_to_group[user.user_id] = group.group_id
                user.status = UserStatus.ASSIGNED
            self._update_queue.put(
                {"type": "group_formed", "group_id": group.group_id})

    def remove_user_from_group(self, user_id: int) -> bool:
        with self._lock:
            if user_id not in self.user_to_group:
                return False
            self._remove_user_from_group(user_id)
            return True

    def get_pending_users(self) -> List[UserLocation]:
        with self._lock:
            return [
                user for user in self.user_locations.values()
                if user.status == UserStatus.PENDING
            ]

    def get_all_users(self) -> List[UserLocation]:
        with self._lock:
            return list(self.user_locations.values())

    def get_group_by_id(self, group_id: str) -> Optional[ClusterGroup]:
        with self._lock:
            return self.cluster_groups.get(group_id)

    def get_group_by_user(self, user_id: int) -> Optional[ClusterGroup]:
        with self._lock:
            if user_id not in self.user_to_group:
                return None
            return self.cluster_groups.get(self.user_to_group[user_id])

    def has_updates(self) -> bool:
        return not self._update_queue.empty()

    def get_update(self, block: bool = True, timeout: float = None) -> Dict:
        try:
            return self._update_queue.get(block=block, timeout=timeout)
        except Empty:
            return {}

    def clear_updates(self):
        with self._lock:
            while not self._update_queue.empty():
                try:
                    self._update_queue.get_nowait()
                except Empty:
                    break

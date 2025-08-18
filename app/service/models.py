from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional, Tuple


@dataclass
class UserLocation:
    user_id: int
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float
    stored_at: datetime

    @property
    def origin_coords(self) -> Tuple[float, float]:
        return (self.origin_lat, self.origin_lng)

    @property
    def destination_coords(self) -> Tuple[float, float]:
        return (self.destination_lat, self.destination_lng)

    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'origin_lat': self.origin_lat,
            'origin_lng': self.origin_lng,
            'destination_lat': self.destination_lat,
            'destination_lng': self.destination_lng,
            'stored_at': self.stored_at.isoformat()
        }


@dataclass
class ClusterGroup:
    group_id: str
    users: List[UserLocation]
    created_at: datetime
    meeting_point_origin: Optional[Tuple[float, float]] = None
    meeting_point_destination: Optional[Tuple[float, float]] = None
    status: str = "forming"  # forming, complete, expired

    def is_complete(self) -> bool:
        return len(self.users) == 3

    def has_user(self, user_id: int) -> bool:
        return any(u.user_id == user_id for u in self.users)

    def get_user_ids(self) -> List[int]:
        return [u.user_id for u in self.users]

    def to_dict(self) -> Dict:
        return {
            'group_id': self.group_id,
            'users': [user.to_dict() for user in self.users],
            'created_at': self.created_at.isoformat(),
            'meeting_point_origin': self.meeting_point_origin,
            'meeting_point_destination': self.meeting_point_destination,
            'status': self.status,
            'user_count': len(self.users)
        }

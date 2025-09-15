from dataclasses import dataclass
from datetime import datetime
from typing import Tuple, Dict, Optional
from enum import Enum

class UserStatus(Enum):
    PENDING = "pending"
    MATCHED = "matched"
    CANCELLED = "cancelled"


@dataclass
class UserLocation:
    user_id: int
    origin_lat: float
    origin_lng: float
    destination_lat: float
    destination_lng: float
    departure_time: datetime
    passengers: int = 1
    # Add origin location details
    origin_city: Optional[str] = None
    origin_county: Optional[str] = None
    origin_state: Optional[str] = None
    # Add destination location details
    destination_city: Optional[str] = None
    destination_county: Optional[str] = None
    destination_state: Optional[str] = None
    status: UserStatus = UserStatus.PENDING

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
            'departure_time': self.departure_time.isoformat(),
            'passengers': self.passengers,
            'origin_city': self.origin_city,
            'origin_county': self.origin_county,
            'origin_state': self.origin_state,
            'destination_city': self.destination_city,
            'destination_county': self.destination_county,
            'destination_state': self.destination_state,
            'status': self.status.value
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            user_id=int(data["user_id"]),
            origin_lat=float(data["origin_lat"]),
            origin_lng=float(data["origin_lng"]),
            destination_lat=float(data["destination_lat"]),
            destination_lng=float(data["destination_lng"]),
            departure_time=datetime.fromisoformat(data["departure_time"]),
            passengers=int(data.get("passengers", 1)),
            origin_city=data.get("origin_city"),
            origin_county=data.get("origin_county"),
            origin_state=data.get("origin_state"),
            destination_city=data.get("destination_city"),
            destination_county=data.get("destination_county"),
            destination_state=data.get("destination_state"),
            status=UserStatus(data.get("status", "pending"))
        )

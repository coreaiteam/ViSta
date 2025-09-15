from geopy.geocoders import Nominatim
from datetime import datetime
from typing import List
from geopy.adapters import AioHTTPAdapter
from .models import UserLocation
import asyncio


### Sync Version ###

# geolocator = Nominatim(user_agent="ride_sharing_app")

# def get_location_info(lat, lon):
#     location = geolocator.reverse((lat, lon), language="en")
#     if location and "address" in location.raw:
#         address = location.raw["address"]
#         return {
#             "city": address.get("city") or address.get("town") or address.get("village"),
#             "county": address.get("county"),
#             "state": address.get("state")
#         }
#     return {"city": None, "county": None, "state": None}

# # Example usage
# info = get_location_info(28.5020, 53.5400)  # somewhere in Jahrom, Fars
# print(info)




### Async Version ###
async def async_get_location_info(geolocator: Nominatim, lat: float, lon: float) -> dict[str, str | None]:
    """
    Fetch city, county, and state info asynchronously from coordinates.
    """
    location = await geolocator.reverse((lat, lon), language="en")
    if location and "address" in location.raw:
        address = location.raw["address"]
        return {
            "city": address.get("city") or address.get("town") or address.get("village"),
            "county": address.get("county"),
            "state": address.get("state"),
        }
    return {"city": None, "county": None, "state": None}

async def build_users() -> List[UserLocation]:
    """
    Build sample dataset with geocoded cities asynchronously.
    """
    coords = [
        (28.91, 50.83, 29.60, 52.53, 1, datetime(2025, 9, 11, 8, 0), 1),
        (28.92, 50.84, 29.61, 52.54, 2, datetime(2025, 9, 11, 8, 15), 2),
        (28.93, 50.85, 29.62, 52.55, 3, datetime(2025, 9, 11, 9, 0), 1),
        (29.60, 52.53, 35.68, 51.41, 4, datetime(2025, 9, 11, 8, 5), 1),
        (29.61, 52.54, 35.69, 51.42, 5, datetime(2025, 9, 11, 8, 20), 1),
    ]

    async with Nominatim(user_agent="my_app", adapter_factory=AioHTTPAdapter) as geolocator:
        tasks = []
        for o_lat, o_lng, d_lat, d_lng, *_ in coords:
            tasks.append(async_get_location_info(geolocator, o_lat, o_lng))
            tasks.append(async_get_location_info(geolocator, d_lat, d_lng))

        results = await asyncio.gather(*tasks, return_exceptions=True)

    # بازسازی یوزرها
    users: List[UserLocation] = []
    for idx, (o_lat, o_lng, d_lat, d_lng, u_id, dep_time, passengers) in enumerate(coords):
        o_info = results[idx * 2]
        d_info = results[idx * 2 + 1]

        # اگر ارور تایم‌اوت باشه، برمی‌گردیم None
        if isinstance(o_info, Exception):
            o_info = {"city": None, "county": None, "state": None}
        if isinstance(d_info, Exception):
            d_info = {"city": None, "county": None, "state": None}

        users.append(
            
            UserLocation(
                user_id=u_id,
                origin_lat=o_lat,
                origin_lng=o_lng,
                destination_lat=d_lat,
                destination_lng=d_lng,
                departure_time=dep_time,
                passengers=passengers,
                origin_city=o_info["city"],
                destination_city=d_info["city"],
            )
        )
    return users
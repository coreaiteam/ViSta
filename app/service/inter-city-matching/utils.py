from datetime import datetime
import asyncio
import math
from typing import List

from geopy.geocoders import Nominatim
from geopy.adapters import AioHTTPAdapter
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

from .models import UserLocation


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


# ----------------------------
# Safe async geocoding helper
# ----------------------------
async def async_get_location_info(
    geolocator: Nominatim, lat: float, lon: float, retries: int = 3, delay: float = 1.0
) -> dict[str, str | None]:
    """
    Fetch city, county, and state info asynchronously from coordinates.
    Retries on timeout/unavailability, with delay between retries.
    """
    for attempt in range(retries):
        try:
            location = await geolocator.reverse((lat, lon), language="en")
            if location and "address" in location.raw:
                address = location.raw["address"]
                return {
                    "city": address.get("city")
                    or address.get("town")
                    or address.get("village"),
                    "county": address.get("county"),
                    "state": address.get("state"),
                }
            return {"city": None, "county": None, "state": None}

        except (GeocoderTimedOut, GeocoderUnavailable):
            if attempt < retries - 1:
                await asyncio.sleep(delay)  # wait before retrying
            else:
                return {"city": None, "county": None, "state": None}


# ----------------------------
# Build sample dataset
# ----------------------------
async def build_users() -> List["UserLocation"]:
    """
    Build sample dataset with geocoded cities asynchronously.
    Handles rate-limits, retries, and graceful fallbacks.
    """
    coords = [
        # (
        #     34.65871450560872,
        #     50.86669921875001,
        #     35.744984905479875,
        #     51.33087158203125,
        #     1,
        #     datetime(2025, 9, 11, 8, 0),
        #     1,
        # ),
        # (
        #     34.630029324737606,
        #     50.85296630859376,
        #     35.79844511798277,
        #     50.99578857421876,
        #     2,
        #     datetime(2025, 9, 11, 8, 15),
        #     2,
        # ),
        (
            29.61855366441621,
            52.51602172851563,
            35.731969883428874,
            51.427001953125,
            3,
            datetime(2025, 9, 11, 9, 0),
            1,
        ),
        (
            29.59820193940507,
            52.49679565429688,
         35.02691940369337, 50.35308837890626,
            4,
            datetime(2025, 9, 11, 9, 5),
            1,
        ),
        # (29.61, 52.54, 35.69, 51.42, 5, datetime(2025, 9, 11, 8, 20), 1),
    ]

    async with Nominatim(
        user_agent="my_app", adapter_factory=AioHTTPAdapter, timeout=5
    ) as geolocator:
        tasks = []
        for o_lat, o_lng, d_lat, d_lng, *_ in coords:
            tasks.append(async_get_location_info(geolocator, o_lat, o_lng))
            tasks.append(async_get_location_info(geolocator, d_lat, d_lng))
        results = await asyncio.gather(*tasks)

    users: List[UserLocation] = []
    for idx, (o_lat, o_lng, d_lat, d_lng, u_id, dep_time, passengers) in enumerate(
        coords
    ):
        o_info = results[idx * 2]
        d_info = results[idx * 2 + 1]

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


def haversine(lat1, lon1, lat2, lon2):
    """Calculate great-circle distance (in km) between two points on Earth."""
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


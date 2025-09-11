from datetime import datetime

from .models import UserLocation
from .matching_service import RideSharingSystem
from .utils import get_location_info

# --- Sample dataset (cities filled by geocoder) ---
users = [
    UserLocation(
        user_id=1,
        origin_lat=28.91,
        origin_lng=50.83,
        destination_lat=29.60,
        destination_lng=52.53,
        departure_time=datetime(2025, 9, 11, 8, 0),
        passengers=1,
        origin_city=get_location_info(28.91, 50.83)["city"],
        destination_city=get_location_info(29.60, 52.53)["city"],
    ),
    UserLocation(
        user_id=2,
        origin_lat=28.92,
        origin_lng=50.84,
        destination_lat=29.61,
        destination_lng=52.54,
        departure_time=datetime(2025, 9, 11, 8, 15),
        passengers=2,
        origin_city=get_location_info(28.92, 50.84)["city"],
        destination_city=get_location_info(29.61, 52.54)["city"],
    ),
    UserLocation(
        user_id=3,
        origin_lat=28.93,
        origin_lng=50.85,
        destination_lat=29.62,
        destination_lng=52.55,
        departure_time=datetime(2025, 9, 11, 9, 0),
        passengers=1,
        origin_city=get_location_info(28.93, 50.85)["city"],
        destination_city=get_location_info(29.62, 52.55)["city"],
    ),
    UserLocation(
        user_id=4,
        origin_lat=29.60,
        origin_lng=52.53,
        destination_lat=35.68,
        destination_lng=51.41,
        departure_time=datetime(2025, 9, 11, 8, 5),
        passengers=1,
        origin_city=get_location_info(29.60, 52.53)["city"],
        destination_city=get_location_info(35.68, 51.41)["city"],
    ),
    UserLocation(
        user_id=5,
        origin_lat=29.61,
        origin_lng=52.54,
        destination_lat=35.69,
        destination_lng=51.42,
        departure_time=datetime(2025, 9, 11, 8, 20),
        passengers=1,
        origin_city=get_location_info(29.61, 52.54)["city"],
        destination_city=get_location_info(35.69, 51.42)["city"],
    ),
]

# --- Run the system ---
system = RideSharingSystem()
for u in users:
    print(f"User {u.user_id} origin={u.origin_city}, destination={u.destination_city}")
    system.add_user(u)

results = system.match_all()

# --- Print results ---
for city, groups in results.items():
    print(f"\nCity: {city}")
    for idx, group in enumerate(groups, 1):
        print(f"  Group {idx}:")
        for u in group:
            print(f"    - User {u.user_id} ({u.passengers} pax) at {u.departure_time}")


"""
TODO:
1- Async call of geocoding API
2- Improve Matching Engine to include nearby cities or cities with route overlap
3- Connent it with the Visualizer.
4- Keep it runinng (Listeninng for new users.)
"""
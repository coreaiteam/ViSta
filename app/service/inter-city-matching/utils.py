from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="ride_sharing_app")

def get_location_info(lat, lon):
    location = geolocator.reverse((lat, lon), language="en")
    if location and "address" in location.raw:
        address = location.raw["address"]
        return {
            "city": address.get("city") or address.get("town") or address.get("village"),
            "county": address.get("county"),
            "state": address.get("state")
        }
    return {"city": None, "county": None, "state": None}

# Example usage
info = get_location_info(28.5020, 53.5400)  # somewhere in Jahrom, Fars
print(info)

#### Appraoch 1: Preload city boundries ####

# import geopandas as gpd
# from shapely.geometry import Point

# # Load shapefile or geojson of city boundaries
# cities = gpd.read_file("geoBoundaries-IRN-ADM4_simplified.geojson", encoding="utf-8")  # replace with your file

# # Ensure coordinate system is WGS84 (lat/lon)
# cities = cities.to_crs(epsg=4326)

# def get_city_from_point(lat, lon):
#     point = Point(lon, lat)  # shapely uses (x, y) = (lon, lat)
#     match = cities[cities.contains(point)]
#     if not match.empty:
#         return match.iloc[0]["shapeName"]  # column name may differ
#     return None

# # Example usage

# print(get_city_from_point(28.910125301654485, 50.832881927490234))# should return "Tehran"


#### Appraoch 2: GeoCoding API ####


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


print(get_location_info(28.910125301654485, 50.832881927490234))

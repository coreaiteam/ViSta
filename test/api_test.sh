# Add a user via API
curl -X POST "http://localhost:8000/users/?user_id=2001&origin_lat=40.7128&origin_lng=-74.0060&destination_lat=34.0522&destination_lng=-118.2437"

# Get all groups
curl "http://localhost:8000/groups/"

# Get service status
curl "http://localhost:8000/status/"
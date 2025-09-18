import json
from datetime import datetime
from pathlib import Path
from ..metrics import evaluate_user_clustering
import dash
from dash import Input, Output, State, html, callback
from dash.exceptions import PreventUpdate
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from datetime import datetime, timezone

from .utils import generate_data, loc2userlocation
from .layouts import main_layout
from .components import map_handler
from app.service.models import UserLocation
from ...service.service import get_clustering_service

# App Initialization
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP])
app.layout = main_layout

# Clustering Service
clustering_service = get_clustering_service()
clustering_service.start()

# User Generator
generated_data = generate_data(clustering_service.clustering_engine.G)
data = iter(generated_data)
num_all_data = len(list(generated_data))   # همه داده‌ها

## Temparary User
temp_user = {"origin": None, "destination": None}

## Temparary Markers
temp_markers = []

## Data Saving
SAVE_DIR = Path("saved_data")
SAVE_DIR.mkdir(exist_ok=True)

    
# def save_users_and_metrics():
#     """Save current users to JSON file and compute metrics"""
#     try:
#         # Get current timestamp for filename
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         # Get all users (as dicts)
#         users_dict = clustering_service.get_all_users()
        
#         if not users_dict:
#             print("No users to save")
#             return None
        
#         # # Save users to JSON
#         # users_filename = SAVE_DIR / f"users_{timestamp}.json"
#         # with open(users_filename, 'w') as f:
#         #     json.dump(users_dict, f, indent=2)
        
#         # print(f"Saved {len(users_dict)} users to {users_filename}")
        
#         # Convert dicts to UserLocation objects for metrics
#         users_objects = []
#         for user_dict in users_dict:
#             try:
#                 user_obj = UserLocation(
#                     user_id=user_dict['user_id'],
#                     origin_lat=user_dict['origin_lat'],
#                     origin_lng=user_dict['origin_lng'],
#                     destination_lat=user_dict['destination_lat'],
#                     destination_lng=user_dict['destination_lng'],
#                     stored_at=datetime.fromisoformat(user_dict['stored_at'].replace('Z', '+00:00')) if isinstance(user_dict['stored_at'], str) else user_dict['stored_at'],
#                     status=user_dict.get('status', 'PENDING')
#                 )
#                 users_objects.append(user_obj)
#             except (KeyError, ValueError) as e:
#                 print(f"Error converting user dict to object: {e}")
#                 continue
        
#         # Get clusters for metrics evaluation - use the clustering engine directly
#         clusters = []
        
#         # Method 1: Try to get clusters from the clustering engine
#         try:
#             # Check if clustering_engine has a way to get current clusters
#             if hasattr(clustering_service.clustering_engine, 'get_current_clusters'):
#                 clusters = clustering_service.clustering_engine.get_current_clusters()
#             elif hasattr(clustering_service.clustering_engine, 'cluster_groups'):
#                 clusters = clustering_service.clustering_engine.cluster_groups
#         except Exception as e:
#             print(f"Error getting clusters from engine: {e}")
        
#         # Method 2: If no clusters from engine, try to cluster the current users
#         if not clusters and users_objects:
#             print("No clusters found from engine, attempting to cluster current users...")
#             try:
#                 # Use the clustering engine to cluster the current users
#                 clusters = clustering_service.clustering_engine.cluster_users(users_objects)
#                 print(f"Clustered {len(users_objects)} users into {len(clusters)} clusters")
#             except Exception as e:
#                 print(f"Error clustering users: {e}")
        
        
#         # Convert clusters to the format expected by metrics evaluation
#         cluster_lists = []
#         if clusters:
#             for cluster in clusters:
#                 cluster_users = []
#                 # Handle different cluster formats
#                 if hasattr(cluster, 'user_ids'):  # ClusterGroup object
#                     for user_id in cluster.user_ids:
#                         user_obj = next((u for u in users_objects if u.user_id == user_id), None)
#                         if user_obj:
#                             cluster_users.append(user_obj)
                
#                 elif hasattr(cluster, 'users'):  # ClusterGroup with users list
#                     for user in cluster.users:
#                         if hasattr(user, 'user_id'):
#                             user_obj = next((u for u in users_objects if u.user_id == user.user_id), None)
#                             if user_obj:
#                                 cluster_users.append(user_obj)
                
#                 elif isinstance(cluster, (list, tuple)):  # List of users
#                     for user in cluster:
#                         if hasattr(user, 'user_id'):
#                             user_obj = next((u for u in users_objects if u.user_id == user.user_id), None)
#                             if user_obj:
#                                 cluster_users.append(user_obj)
                
#                 elif isinstance(cluster, dict):  # Dictionary format
#                     user_ids = cluster.get('user_ids', [])
#                     for user_id in user_ids:
#                         user_obj = next((u for u in users_objects if u.user_id == user_id), None)
#                         if user_obj:
#                             cluster_users.append(user_obj)
                
#                 if cluster_users:
#                     cluster_lists.append(cluster_users)
        
#         # Evaluate metrics if we have clusters
#         if cluster_lists:
#             print(f"Found {len(cluster_lists)} clusters for metrics evaluation")
            
#             # Get the graph from clustering service
#             graph = clustering_service.clustering_engine.G
            
#             metrics = evaluate_user_clustering(
#                 user_locations=users_objects,
#                 clusters=cluster_lists,
#                 graph=graph,
#                 alpha=1.0
#             )
            
#             # Save metrics to JSON
#             metrics_filename = SAVE_DIR / f"metrics_{clustering_service.engine_name}_{timestamp}.json"
#             with open(metrics_filename, 'w') as f:
#                 json.dump(metrics.to_dict(), f, indent=2)
            
#             print(f"Saved metrics to {metrics_filename}")
            
#             # Print summary
#             print(f"\n=== METRICS SUMMARY ===")
#             print(f"Silhouette Score: {metrics.combined_silhouette:.3f}")
#             print(f"Dunn Index: {metrics.dun_index:.3f}")
#             print(f"Combined SSE: {metrics.combined_sse:.2f}")
#             print(f"Total Users: {len(users_objects)}")
#             print(f"Total Clusters: {len(cluster_lists)}")
            
#             # Print cluster sizes
#             print(f"Cluster sizes: {[len(cluster) for cluster in cluster_lists]}")
            
#             return metrics
#         else:
#             print("No valid clusters found for metrics evaluation")
#             print(f"Users available: {len(users_objects)}")
#             if users_objects:
#                 print("Creating single cluster with all users for basic metrics...")
#                 # Create one big cluster with all users for basic metrics
#                 cluster_lists = [users_objects]
                
#                 # Get the graph from clustering service
#                 graph = clustering_service.clustering_engine.G
                
#                 metrics = evaluate_user_clustering(
#                     user_locations=users_objects,
#                     clusters=cluster_lists,
#                     graph=graph,
#                     alpha=1.0
#                 )
                
#                 # Save metrics to JSON
#                 metrics_filename = SAVE_DIR / f"metrics_{timestamp}_single_cluster.json"
#                 with open(metrics_filename, 'w') as f:
#                     json.dump(metrics.to_dict(), f, indent=2)
                
#                 print(f"Saved single-cluster metrics to {metrics_filename}")
#                 print(f"Single cluster metrics - Silhouette: {metrics.combined_silhouette:.3f}")
                
#                 return metrics
        
#         return None
        
#     except Exception as e:
#         print(f"Error saving users and metrics: {e}")
#         import traceback
#         traceback.print_exc()
#         return None

def save_users_and_metrics():
    """Save current users to JSON file and compute metrics"""
    try:
        # Get current timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get all users (as dicts)
        users_dict = clustering_service.get_all_users()
        
        if not users_dict:
            print("No users to save")
            return None
        
        # # Save users to JSON
        # users_filename = SAVE_DIR / f"users_{timestamp}.json"
        # with open(users_filename, 'w') as f:
        #     json.dump(users_dict, f, indent=2)
        
        # print(f"Saved {len(users_dict)} users to {users_filename}")
        
        # Convert dicts to UserLocation objects for metrics
        users_objects = []
        for user_dict in users_dict:
            try:
                user_obj = UserLocation(
                    user_id=user_dict['user_id'],
                    origin_lat=user_dict['origin_lat'],
                    origin_lng=user_dict['origin_lng'],
                    destination_lat=user_dict['destination_lat'],
                    destination_lng=user_dict['destination_lng'],
                    stored_at=datetime.fromisoformat(user_dict['stored_at'].replace('Z', '+00:00')) if isinstance(user_dict['stored_at'], str) else user_dict['stored_at'],
                    status=user_dict.get('status', 'PENDING')
                )
                users_objects.append(user_obj)
            except (KeyError, ValueError) as e:
                print(f"Error converting user dict to object: {e}")
                continue
        
        # Get clusters for metrics evaluation
        clusters = clustering_service.get_all_active_groups()
        print(clusters)
        if not clusters:
            print("No clusters found from service")
            return None
        
        print(f"Found {len(clusters)} cluster groups from service")
        
        # Convert clusters to the format expected by metrics evaluation
        cluster_lists = []
        meeting_points = {}  # {cluster_id: {'origin': (lat, lng), 'destination': (lat, lng)}}
        
        for cluster_idx, cluster_dict in enumerate(clusters):
            cluster_users = []
            
            # Extract users from the cluster dictionary
            if 'users' in cluster_dict and isinstance(cluster_dict['users'], list):
                for user_dict in cluster_dict['users']:
                    # Find the corresponding UserLocation object
                    user_obj = next((u for u in users_objects if u.user_id == user_dict['user_id']), None)
                    if user_obj:
                        cluster_users.append(user_obj)
            
            if cluster_users:
                cluster_lists.append(cluster_users)
                
                # Extract meeting points from algorithm
                if 'meeting_point_origin' in cluster_dict and 'meeting_point_destination' in cluster_dict:
                    try:
                        # Convert numpy types to regular Python floats
                        origin_mp = cluster_dict['meeting_point_origin']
                        dest_mp = cluster_dict['meeting_point_destination']
                        
                        meeting_points[cluster_idx] = {
                            'origin': (float(origin_mp[0]), float(origin_mp[1])),
                            'destination': (float(dest_mp[0]), float(dest_mp[1]))
                        }
                        print(f"Cluster {cluster_idx}: using algorithm meeting points")
                    except (TypeError, IndexError) as e:
                        print(f"Error extracting meeting points for cluster {cluster_idx}: {e}")
                        # Fallback to calculated centroids
                
                print(f"Cluster {cluster_idx}: {len(cluster_users)} users")
        
        # Evaluate metrics if we have clusters
        if cluster_lists:
            print(f"Prepared {len(cluster_lists)} clusters for metrics evaluation")
            
            # Get the graph from clustering service
            graph = clustering_service.clustering_engine.G
            
            # Use algorithm meeting points if available, otherwise calculate
            centroid_method = "algorithm" if meeting_points else "calculate"
            
            metrics = evaluate_user_clustering(
                user_locations=users_objects,
                clusters=cluster_lists,
                graph=graph,
                meeting_points=meeting_points,
                alpha=1.0,
                centroid_method=centroid_method
            )
        # cluster_lists = []
        
        # for cluster_dict in clusters:
        #     cluster_users = []
            
        #     # Extract users from the cluster dictionary
        #     if 'users' in cluster_dict and isinstance(cluster_dict['users'], list):
        #         for user_dict in cluster_dict['users']:
        #             # Find the corresponding UserLocation object
        #             user_obj = next((u for u in users_objects if u.user_id == user_dict['user_id']), None)
        #             if user_obj:
        #                 cluster_users.append(user_obj)
        #             else:
        #                 # If user object not found, create one from the user dict
        #                 try:
        #                     new_user_obj = UserLocation(
        #                         user_id=user_dict['user_id'],
        #                         origin_lat=user_dict['origin_lat'],
        #                         origin_lng=user_dict['origin_lng'],
        #                         destination_lat=user_dict['destination_lat'],
        #                         destination_lng=user_dict['destination_lng'],
        #                         stored_at=datetime.fromisoformat(user_dict['stored_at'].replace('Z', '+00:00')) if isinstance(user_dict['stored_at'], str) else user_dict['stored_at'],
        #                         status=user_dict.get('status', 'PENDING')
        #                     )
        #                     cluster_users.append(new_user_obj)
        #                 except (KeyError, ValueError) as e:
        #                     print(f"Error creating user object from cluster data: {e}")
        #                     continue
            
        #     if cluster_users:
        #         cluster_lists.append(cluster_users)
        #         # print(f"Cluster {cluster_dict.get('group_id', 'unknown')}: {len(cluster_users)} users")
        
        # # Evaluate metrics if we have clusters
        # if cluster_lists:
        #     print(f"Prepared {len(cluster_lists)} clusters for metrics evaluation")
            
        #     # Get the graph from clustering service
        #     graph = clustering_service.clustering_engine.G
            
        #     metrics = evaluate_user_clustering(
        #         user_locations=users_objects,
        #         clusters=cluster_lists,
        #         graph=graph,
        #         alpha=1.0
        #     )
            
            # Save metrics to JSON
            metrics_filename = SAVE_DIR / f"metrics_{clustering_service.engine_name}_{timestamp}.json"
            with open(metrics_filename, 'w') as f:
                json.dump(metrics.to_dict(), f, indent=2)
            
            print(f"Saved metrics to {metrics_filename}")
            
            # Print summary
            print(f"\n=== METRICS SUMMARY ===")
            print(f"Silhouette Score: {metrics.combined_silhouette:.3f}")
            print(f"Dunn Index: {metrics.dun_index:.3f}")
            print(f"Combined SSE: {metrics.combined_sse:.2f}")
            print(f"Origin SSE: {metrics.origin_metrics.sse:.2f}")
            print(f"Destination SSE: {metrics.destination_metrics.sse:.2f}")
            print(f"Total Users: {len(users_objects)}")
            print(f"Total Clusters: {len(cluster_lists)}")
            
            # Print cluster sizes
            cluster_sizes = [len(cluster) for cluster in cluster_lists]
            print(f"Cluster sizes: {cluster_sizes}")
            print(f"Average cluster size: {sum(cluster_sizes) / len(cluster_sizes):.2f}")
            
            # Add meeting point info to metrics
            metrics_dict = metrics.to_dict()
            metrics_dict['centroid_method'] = centroid_method
            metrics_dict['algorithm_meeting_points_used'] = bool(meeting_points)
            
            return metrics
        else:
            print("No valid clusters could be prepared for metrics evaluation")
        
        return None
        
    except Exception as e:
        print(f"Error saving users and metrics: {e}")
        import traceback
        traceback.print_exc()
        return None

# Add this callback to save data
@callback(
    Output("save-status", "children"),
    Input("save-data-btn", "n_clicks"),
    prevent_initial_call=True,
)
def save_data(n_clicks):
    """Callback to save current users and compute metrics"""
    if not n_clicks:
        raise PreventUpdate
    
    result = save_users_and_metrics()
    
    if result:
        return dbc.Alert("Data saved successfully!", color="success")
    else:
        return dbc.Alert("Failed to save data or no data available", color="warning")
    
@callback(
    Output("temp-markers", "children", allow_duplicate=True),
    [
        Input("main-map", "clickData"),
        State("click-mode", "value"),
    ],
    prevent_initial_call=True,
)
def capture_map_click(click_data, click_mode):
    """
    Callback to capture map clicks and update temporary coordinates with markers.

    Args:
        click_data: Data from map click event
        click_mode: Current click mode (origin/destination)
        temp_coords: Temporary coordinates stored in the app

    Returns:
        Tuple of updated coordinates and updated map with temporary markers
    """
    if not click_data:
        raise PreventUpdate

    # Extract latitude and longitude from click event
    lat, lng = click_data["latlng"]["lat"], click_data["latlng"]["lng"]
    coords = [lat, lng]

    # Update temp_coords based on click_mode
    global temp_user
    temp_user[click_mode] = None
    temp_user[click_mode] = coords


    # Create temporary markers

    global temp_markers

    if click_mode == "origin":
        temp_markers = []
    
    if click_mode == "destination":
        temp_markers.pop()

    for point_type, position in [
        ("Origin", temp_user.get("origin")),
        ("Destination", temp_user.get("destination")),
    ]:
        if position:
            temp_markers.append(
                dl.Marker(
                    position=position,
                    children=[
                        dl.Tooltip(f"Selected {point_type} (Temporary)"),
                        dl.Popup(
                            [
                                html.H4(f"Selected {point_type}"),
                                html.P(f"Coordinates: {position}"),
                            ]
                        ),
                    ],
                )
            )

    return temp_markers


@callback(
    Output("users", "children", allow_duplicate=True),
    Output("clusters", "children", allow_duplicate=True),
    Output("bulk-user-info", "children", allow_duplicate=True),

    Input("add-random-user-btn", "n_clicks"),
    prevent_initial_call=True,
)
def add_random_user(n_clicks):
    """
    Callback to add a new random user to the map.

    Args:
        n_clicks: Number of clicks on random user button

    Returns:
        Updated map with new random user
    """
    if not n_clicks:
        raise PreventUpdate

    user_id = clustering_service.get_next_user_id()
    
    new_user = loc2userlocation(user_id=user_id, loc=next(data))
    clustering_service.add_user_location(new_user)

    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()
    userlayer, cluster_layer =  map_handler.create_users_and_clusters_layer(users=users, clusters=clusters)

    global num_all_data
    num_all_data -= 1
    info_text = f"Remaining number of users: {num_all_data}"

    return userlayer, cluster_layer, info_text


@callback(
    Output("users", "children", allow_duplicate=True),
    Output("clusters", "children", allow_duplicate=True),
    Output("temp-markers", "children", allow_duplicate=True),


    Input("add-selected-user-btn", "n_clicks"),
    prevent_initial_call=True,
)
def add_selected_user(n_clicks):
    """
    Callback to add a user with selected coordinates.

    Args:
        n_clicks: Number of clicks on selected user button
        temp_coords: Temporary coordinates for selected user

    Returns:
        Updated map with new selected user
    """
    if not n_clicks:
        raise PreventUpdate

    global temp_user
    if not (temp_user.get("origin") and temp_user.get("destination")):
        raise PreventUpdate

    new_user = UserLocation.from_dict(
        {
            "user_id": clustering_service.get_next_user_id(),
            "origin_lat": temp_user["origin"][0],
            "origin_lng": temp_user["origin"][1],
            "destination_lat": temp_user["destination"][0],
            "destination_lng": temp_user["destination"][1],
            "stored_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    temp_user = {"origin": None, "destination": None}
    global temp_markers
    temp_markers = []
    clustering_service.add_user_location(new_user)

    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()
    user_layer, cluster_layer = map_handler.create_users_and_clusters_layer(users=users, clusters=clusters)
    return user_layer, cluster_layer, temp_markers


@callback(
    Output("users", "children", allow_duplicate=True),
    Output("clusters", "children", allow_duplicate=True),
    Input("users-refresh-interval", "n_intervals"),
    prevent_initial_call=True,
)
def refresh_map(n_intervals):
    """
    Periodically refresh the map with updated users and clusters.
    """
    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()

    user_layer, cluster_layer = map_handler.create_users_and_clusters_layer(
        users=users,
        clusters=clusters
    )
    return user_layer, cluster_layer


@callback(
    Output("temp-markers", "children", allow_duplicate=True),
    Input("clear-markers-btn", "n_clicks"),
    prevent_initial_call=True,
)
def clear_temp_markers(n_clicks):
    """
    Clears all temporary markers from the map.
    """
    global temp_user, temp_markers
    temp_user = {"origin": None, "destination": None}
    temp_markers = []
    return []


@callback(
    Output("stats-container", "children"),
    Input("stats-refresh-interval", "n_intervals"),  # reuse the interval you already have
    prevent_initial_call=True,
)
def update_stats(n_intervals):
    """
    Updates the statistics card with the latest user and cluster counts.
    """
    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()

    total_users = len(users)
    total_clusters = len(clusters)

    return dbc.Row(
        [
            dbc.Col(
                html.Div(
                    [
                        html.H5("Total Users", className="text-muted"),
                        html.H3(total_users, className="text-success fw-bold"),
                    ],
                    className="text-center",
                ),
                width=6,
            ),
            dbc.Col(
                html.Div(
                    [
                        html.H5("Total Clusters", className="text-muted"),
                        html.H3(total_clusters, className="text-primary fw-bold"),
                    ],
                    className="text-center",
                ),
                width=6,
            ),
        ],
        className="g-3",
    )


@callback(
    [
        Output("users", "children", allow_duplicate=True),
        Output("clusters", "children", allow_duplicate=True),
    ],
    Input("remove-user-btn", "n_clicks"),
    State("remove-user-id", "value"),
    prevent_initial_call=True,
)
def remove_user(n_clicks, user_id):
    """
    Removes a user by user_id and refreshes map
    """
    if not user_id:
        raise PreventUpdate

    print(f"user id : {user_id}")
    
    clustering_service.remove_user(user_id)

    # refresh data
    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()
    user_layer, cluster_layer = map_handler.create_users_and_clusters_layer(users, clusters)

    return user_layer, cluster_layer



@callback(
    [
        Output("users", "children", allow_duplicate=True),
        Output("clusters", "children", allow_duplicate=True),
        Output("bulk-user-info", "children"),
    ],
    Input("add-multiple-users-btn", "n_clicks"),
    State("bulk-user-count", "value"),
    prevent_initial_call=True,
)
def add_multiple_users(n_clicks, count):
    """
    Add multiple random users at once.
    """
    if not n_clicks or not count or count <= 0:
        raise PreventUpdate
    global num_all_data
    users_added = 0

    for _ in range(count):

        user_id = clustering_service.get_next_user_id()
        new_user = loc2userlocation(user_id=user_id, loc=next(data))
        clustering_service.add_user_location(new_user)
        users_added += 1

    # refresh data
    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()
    user_layer, cluster_layer = map_handler.create_users_and_clusters_layer(users, clusters)
    num_all_data -= users_added
    info_text = f"Remaining number of users: {num_all_data}"

    return user_layer, cluster_layer, info_text



if __name__ == "__main__":
    app.run(port=8080)


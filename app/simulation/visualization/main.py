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


import time
from threading import Lock
import dash
from dash import Input, Output, State, html, callback, callback_context, MATCH
from dash.exceptions import PreventUpdate
import dash_leaflet as dl
import dash_bootstrap_components as dbc
from datetime import datetime, timezone

from .utils import generate_data, loc2userlocation
from .layouts import main_layout, intra_map_handler, inter_map_handler
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
num_all_data = {
    "intra": len(list(generated_data)),
    "inter": 100,
}  # مثال: 100 داده اولیه برای هرکدام


# نگهداری وضعیت کاربران موقت برای هر prefix
temp_users = {
    "intra": {"origin": None, "destination": None},
    "inter": {"origin": None, "destination": None},
}

# نگهداری مارکرهای موقت برای هر prefix
temp_markers = {"intra": [], "inter": []}


## For loading Map Tile well
@app.callback(
    Output({"type": "main-map", "prefix": "intra"}, "invalidateSize"),
    Input("clustering-tabs", "active_tab"),
)
def invalidate_intra_map(active_tab):
    if active_tab == "intra-city":
        return datetime.now().isoformat()  # Any changing value triggers the invalidate
    raise PreventUpdate

@app.callback(
    Output({"type": "main-map", "prefix": "inter"}, "invalidateSize"),
    Input("clustering-tabs", "active_tab"),
)
def invalidate_inter_map(active_tab):
    if active_tab == "inter-city":
        return datetime.now().isoformat()  # Any changing value triggers the invalidate
    raise PreventUpdate


@callback(
    Output({"type": "temp-markers", "prefix": MATCH}, "children", allow_duplicate=True),
    [
        Input({"type": "main-map", "prefix": MATCH}, "clickData"),
        State({"type": "click-mode", "prefix": MATCH}, "value"),
    ],
    prevent_initial_call=True,
)
def capture_map_click(click_data, click_mode):
    """
    Capture map clicks and update temporary coordinates with markers.
    Works for both intra-city and inter-city maps (pattern-matching).
    """
    if not click_data:
        raise PreventUpdate

    # مشخص کردن prefix (intra یا inter)
    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    # Extract latitude and longitude
    lat, lng = click_data["latlng"]["lat"], click_data["latlng"]["lng"]
    coords = [lat, lng]

    # Update temp_user for this prefix
    temp_users[prefix][click_mode] = None
    temp_users[prefix][click_mode] = coords

    # Update markers
    if click_mode == "origin":
        temp_markers[prefix] = []

    if click_mode == "destination" and temp_markers[prefix]:
        temp_markers[prefix].pop()

    for point_type, position in [
        ("Origin", temp_users[prefix].get("origin")),
        ("Destination", temp_users[prefix].get("destination")),
    ]:
        if position:
            temp_markers[prefix].append(
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

    return temp_markers[prefix]


@callback(
    Output({"type": "users", "prefix": MATCH}, "children", allow_duplicate=True),
    Output({"type": "clusters", "prefix": MATCH}, "children", allow_duplicate=True),
    Output({"type": "temp-markers", "prefix": MATCH}, "children", allow_duplicate=True),
    Input({"type": "add-selected-user-btn", "prefix": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def add_selected_user(n_clicks):
    """
    Add a user with the currently selected temporary coordinates (origin + destination).
    Works for both intra-city and inter-city layouts using prefix.
    """
    if not n_clicks:
        raise PreventUpdate

    # Extract prefix from triggering input
    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    # Check that both points exist
    if not (temp_users[prefix].get("origin") and temp_users[prefix].get("destination")):
        raise PreventUpdate

    # Generate layers (map_handler خودش باید prefix-aware ساخته شده باشه)
    if prefix == "intra":
        # Create new user from selected coordinates
        new_user = UserLocation.from_dict(
            {
                "user_id": clustering_service.get_next_user_id(),
                "origin_lat": temp_users[prefix]["origin"][0],
                "origin_lng": temp_users[prefix]["origin"][1],
                "destination_lat": temp_users[prefix]["destination"][0],
                "destination_lng": temp_users[prefix]["destination"][1],
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }
        )

        # Update clustering service
        clustering_service.add_user_location(new_user)
        users = clustering_service.get_all_users()
        clusters = clustering_service.get_all_active_groups()
        user_layer = intra_map_handler.create_users_layer(
            users=users, clusters=clusters
        )
        cluster_layer = intra_map_handler.create_clusters_layer(clusters=clusters)

    else:
        pass

    # Reset temp data for this prefix
    temp_users[prefix]["origin"] = None
    temp_users[prefix]["destination"] = None
    temp_markers[prefix].clear()

    return user_layer, cluster_layer, temp_markers


@callback(
    Output({"type": "temp-markers", "prefix": MATCH}, "children", allow_duplicate=True),
    Input({"type": "clear-markers-btn", "prefix": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def clear_temp_markers(n_clicks):
    """
    Clears all temporary markers and resets temp coordinates for the given prefix.
    """
    if not n_clicks:
        raise PreventUpdate

    # تشخیص prefix از triggering input
    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    # ریست کردن داده‌های موقت برای این prefix
    temp_users[prefix]["origin"] = None
    temp_users[prefix]["destination"] = None
    temp_markers[prefix].clear()

    return []


# شمارنده دیتای باقی‌مانده برای هر prefix


@callback(
    Output({"type": "users", "prefix": MATCH}, "children", allow_duplicate=True),
    Output({"type": "clusters", "prefix": MATCH}, "children", allow_duplicate=True),
    Output(
        {"type": "bulk-user-info", "prefix": MATCH}, "children", allow_duplicate=True
    ),
    Input({"type": "add-random-user-btn", "prefix": MATCH}, "n_clicks"),
    prevent_initial_call=True,
)
def add_random_user(n_clicks):
    """
    Add a new random user to the map (currently only intra-city implemented).
    """
    if not n_clicks:
        raise PreventUpdate

    # گرفتن prefix از input
    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    if prefix == "intra":
        # ساخت یوزر جدید
        user_id = clustering_service.get_next_user_id()
        new_user = loc2userlocation(user_id=user_id, loc=next(data))
        clustering_service.add_user_location(new_user)

        # گرفتن لیست کاربران و کلاسترها
        users = clustering_service.get_all_users()
        clusters = clustering_service.get_all_active_groups()

        # ساخت لایه‌های نقشه با intra_map_handler
        userlayer = intra_map_handler.create_users_layer(users=users, clusters=clusters)
        cluster_layer = intra_map_handler.create_clusters_layer(clusters=clusters)

        # آپدیت شمارنده
        global num_all_data
        num_all_data["intra"] -= 1
        info_text = f"Remaining number of users: {num_all_data['intra']}"

        return userlayer, cluster_layer, info_text

    elif prefix == "inter":
        # TODO: بعداً منطق inter-city اضافه میشه
        raise PreventUpdate


# Global cache per prefix
last_users = {"intra": None, "inter": None}
last_clusters = {"intra": None, "inter": None}

# Lock per prefix
callback_locks = {"intra": Lock(), "inter": Lock()}

# Last execution time per prefix
last_execution_time = {"intra": 0, "inter": 0}
EXECUTION_COOLDOWN = 0.5  # 500ms cooldown


@callback(
    Output({"type": "users", "prefix": MATCH}, "children", allow_duplicate=True),
    Output({"type": "clusters", "prefix": MATCH}, "children", allow_duplicate=True),
    Input({"type": "users-refresh-interval", "prefix": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def refresh_map(n_intervals):
    """
    Periodically refresh the map with updated users and clusters.
    Currently implemented only for intra prefix.
    """
    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    if prefix == "intra":
        global last_users, last_clusters, last_execution_time, callback_locks

        current_time = time.time()
        if current_time - last_execution_time[prefix] < EXECUTION_COOLDOWN:
            print(
                f"[{prefix}] Skipping execution - too soon after last run: {current_time - last_execution_time[prefix]:.3f}s"
            )
            raise PreventUpdate

        with callback_locks[prefix]:
            if current_time - last_execution_time[prefix] < EXECUTION_COOLDOWN:
                raise PreventUpdate

            last_execution_time[prefix] = current_time

            users = clustering_service.get_all_users()
            clusters = clustering_service.get_all_active_groups()

            # Skip if nothing changed
            if users == last_users[prefix] and clusters == last_clusters[prefix]:
                raise PreventUpdate

            # Update cache
            last_users[prefix] = users
            last_clusters[prefix] = clusters

            user_layer = intra_map_handler.create_users_layer(
                users=users, clusters=clusters
            )
            cluster_layer = intra_map_handler.create_clusters_layer(clusters=clusters)

            return user_layer, cluster_layer
    else:
        # prefix = "inter" or others -> do nothing
        raise PreventUpdate


@callback(
    Output({"type": "stats-container", "prefix": MATCH}, "children"),
    Input({"type": "stats-refresh-interval", "prefix": MATCH}, "n_intervals"),
    prevent_initial_call=True,
)
def update_stats(n_intervals):
    """
    Updates the statistics card with the latest user and cluster counts.
    Currently implemented only for intra prefix.
    """
    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    if prefix == "intra":
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
    else:
        # prefix = inter or others → فعلاً کاری انجام نمی‌ده
        raise PreventUpdate


@callback(
    [
        Output({"type": "users", "prefix": MATCH}, "children", allow_duplicate=True),
        Output({"type": "clusters", "prefix": MATCH}, "children", allow_duplicate=True),
    ],
    Input({"type": "remove-user-btn", "prefix": MATCH}, "n_clicks"),
    State({"type": "remove-user-id", "prefix": MATCH}, "value"),
    prevent_initial_call=True,
)
def remove_user(n_clicks, user_id):
    """
    Removes a user by user_id and refreshes the map for the given prefix.
    Currently implemented only for intra.
    """
    if not n_clicks or not user_id:
        raise PreventUpdate

    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    if prefix == "intra":
        clustering_service.remove_user(user_id)

        # refresh data
        users = clustering_service.get_all_users()
        clusters = clustering_service.get_all_active_groups()

        user_layer = intra_map_handler.create_users_layer(
            users=users, clusters=clusters
        )
        cluster_layer = intra_map_handler.create_clusters_layer(clusters=clusters)

        return user_layer, cluster_layer
    else:
        # prefix = inter یا بقیه → فعلاً کاری انجام نمی‌ده
        raise PreventUpdate


@callback(
    [
        Output({"type": "users", "prefix": MATCH}, "children", allow_duplicate=True),
        Output({"type": "clusters", "prefix": MATCH}, "children", allow_duplicate=True),
        Output({"type": "bulk-user-info", "prefix": MATCH}, "children"),
    ],
    Input({"type": "add-multiple-users-btn", "prefix": MATCH}, "n_clicks"),
    State({"type": "bulk-user-count", "prefix": MATCH}, "value"),
    prevent_initial_call=True,
)
def add_multiple_users(n_clicks, count):
    """
    Add multiple random users at once for the given prefix.
    Currently implemented only for intra.
    """
    if not n_clicks or not count or count <= 0:
        raise PreventUpdate

    prefix = callback_context.inputs_list[0]["id"]["prefix"]

    if prefix == "intra":
        users_added = 0

        for _ in range(count):
            user_id = clustering_service.get_next_user_id()
            new_user = loc2userlocation(user_id=user_id, loc=next(data))
            clustering_service.add_user_location(new_user)
            users_added += 1

        # refresh data
        users = clustering_service.get_all_users()
        clusters = clustering_service.get_all_active_groups()

        user_layer = intra_map_handler.create_users_layer(
            users=users, clusters=clusters
        )
        cluster_layer = intra_map_handler.create_clusters_layer(clusters=clusters)

        # کم کردن از شمارنده داده‌های باقی‌مانده
        num_all_data[prefix] -= users_added
        info_text = f"Remaining number of users: {num_all_data[prefix]}"

        return user_layer, cluster_layer, info_text
    else:
        # prefix = inter یا بقیه → فعلاً کاری انجام نمی‌ده
        raise PreventUpdate


if __name__ == "__main__":
    app.run(port=8080)

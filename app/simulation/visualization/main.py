import dash
from dash import Input, Output, State, html
from dash.exceptions import PreventUpdate
from dash import Dash, Input, Output, State, callback  # noqa: F811
from datetime import datetime, timezone
import dash_leaflet as dl
import dash_bootstrap_components as dbc


from .utils import generate_data, loc2userlocation
from .layouts import main_layout, map_handler
from app.service.models import UserLocation


from ...service.service import get_clustering_service


## App Initialization
app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY, dbc.icons.BOOTSTRAP])
app.layout = main_layout

## Clustering Service
clustering_service = get_clustering_service()
clustering_service.start()

## User Generator
data = generate_data()
data = iter(data)

## Users

# Callback to capture map clicks and update temporary coordinates
@callback(
    Output("temp-coordinates", "data"),
    Output("map-container", "children", allow_duplicate=True),
    Input("main-map", "clickData"),
    State("click-mode", "value"),
    State("temp-coordinates", "data"),

    prevent_initial_call=True,
)
def capture_map_click(
    click_data, click_mode, temp_coords,
):
    if not click_data:
        raise PreventUpdate

    # Extract latitude and longitude from click event
    lat, lng = click_data["latlng"]["lat"], click_data["latlng"]["lng"]
    coords = [lat, lng]

    # Update temp_coords based on click_mode
    updated_coords = temp_coords.copy()
    updated_coords[click_mode] = coords

    # Add temporary markers for Origin
    temp_markers = []
    if updated_coords["origin"]:
        temp_markers.append(
            dl.Marker(
                position=updated_coords["origin"],
                children=[
                    dl.Tooltip("Selected Origin (Temporary)"),
                    dl.Popup(
                        [
                            html.H4("Selected Origin"),
                            html.P(f"Coordinates: {updated_coords['origin']}"),
                        ]
                    ),
                ],
            )
        )

    # Add temporary markers for Destination
    if updated_coords["destination"]:
        temp_markers.append(
            dl.Marker(
                position=updated_coords["destination"],
                children=[
                    dl.Tooltip("Selected Destination (Temporary)"),
                    dl.Popup(
                        [
                            html.H4("Selected Destination"),
                            html.P(f"Coordinates: {updated_coords['destination']}"),
                        ]
                    ),
                ],
            )
        )

    users = clustering_service.get_all_users()
    clusters = clustering_service.get_all_active_groups()
    updated_map = map_handler.create_map(
        users=users, components=temp_markers, clusters=clusters
    )

    return (updated_coords, updated_map)


# Callback to add new users (random or selected)
@callback(
    Output("map-container", "children", allow_duplicate=True),
    Input("add-random-user-btn", "n_clicks"),
    Input("add-selected-user-btn", "n_clicks"),
    State("temp-coordinates", "data"),
    prevent_initial_call=True,
)
def add_new_user(random_btn_clicks, selected_btn_clicks, temp_coords):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]

    if triggered_id == "add-random-user-btn" and random_btn_clicks > 0:
        # Generate and add a random user

        # new_user = UserLocation.from_dict(generate_random_users(
        #     n=1, origin_center=(35.795, 51.435), destination_center=(35.650, 51.380)
        # ))
        user_id = len(clustering_service.get_all_users()) + 1
        new_user = loc2userlocation(user_id=user_id, loc=next(data))

        clustering_service.add_user_location(new_user)
        users = clustering_service.get_all_users()
        clusters = clustering_service.get_all_active_groups()

        # new_user[0]["user_id"] = len(current_users) + 1
        # updated_users = current_users + new_user
        updated_map = map_handler.create_map(users=users, clusters=clusters)
        return updated_map

    elif triggered_id == "add-selected-user-btn" and selected_btn_clicks > 0:
        # Add user with selected origin/destination
        if temp_coords["origin"] is None or temp_coords["destination"] is None:
            raise PreventUpdate  # Don't add if both coordinates aren't set

        new_user = UserLocation.from_dict(
            {
                "user_id": len(clustering_service.get_all_users()) + 1,  # Incremental integer ID
                "origin_lat": temp_coords["origin"][0],
                "origin_lng": temp_coords["origin"][1],
                "destination_lat": temp_coords["destination"][0],
                "destination_lng": temp_coords["destination"][1],
                "stored_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        clustering_service.add_user_location(new_user)
        users = clustering_service.get_all_users()
        clusters = clustering_service.get_all_active_groups()
        # Create a new user matching the UserLocation.to_dict() structure

        # updated_users = current_users + new_user
        updated_map = map_handler.create_map(users=users, clusters=clusters)
        return updated_map

    raise PreventUpdate




if __name__ == "__main__":
    app.run(port="8080")

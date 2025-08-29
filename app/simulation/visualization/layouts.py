from datetime import datetime
import math
from typing import List, Dict, Optional, Tuple
import uuid
import dash_bootstrap_components as dbc

from dash import html, dcc  # noqa: F811
from dash_extensions import WebSocket

from .components import map_handler, UserLocation
from .utils import generate_random_users


from app.service.models import ClusterGroup


# Initialize with some users
# initial_users = generate_random_users(
#     n=2, origin_center=(35.795, 51.435), destination_center=(35.650, 51.380)
# )


# def calculate_distance(
#     coord1: Tuple[float, float], coord2: Tuple[float, float]
# ) -> float:
#     """Calculate approximate distance between two coordinates (in degrees)"""
#     lat1, lng1 = coord1
#     lat2, lng2 = coord2
#     return math.sqrt((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2)


# def create_clusters(users: List[Dict]) -> List[Dict]:
#     """Create clusters from a list of user dictionaries"""
#     user_objects = [UserLocation.from_dict(user) for user in users]
#     clusters = []
#     used_user_ids = set()
#     threshold = 0.02  # Distance threshold for clustering (in degrees)

#     for user in user_objects:
#         if user.user_id in used_user_ids:
#             continue

#         # Start a new cluster
#         cluster_users = [user]
#         used_user_ids.add(user.user_id)

#         # Find other users with nearby origins and destinations
#         for other_user in user_objects:
#             if other_user.user_id in used_user_ids or len(cluster_users) >= 3:
#                 continue
#             origin_dist = calculate_distance(
#                 user.origin_coords, other_user.origin_coords
#             )
#             dest_dist = calculate_distance(
#                 user.destination_coords, other_user.destination_coords
#             )
#             if origin_dist < threshold and dest_dist < threshold:
#                 cluster_users.append(other_user)
#                 used_user_ids.add(other_user.user_id)

#         # Calculate meeting points as average of origins and destinations
#         if len(cluster_users) > 1:  # Only create clusters with 2+ users
#             avg_origin_lat = sum(u.origin_lat for u in cluster_users) / len(
#                 cluster_users
#             )
#             avg_origin_lng = sum(u.origin_lng for u in cluster_users) / len(
#                 cluster_users
#             )
#             avg_dest_lat = sum(u.destination_lat for u in cluster_users) / len(
#                 cluster_users
#             )
#             avg_dest_lng = sum(u.destination_lng for u in cluster_users) / len(
#                 cluster_users
#             )

#             cluster = ClusterGroup(
#                 group_id=str(uuid.uuid4()),
#                 users=cluster_users,
#                 created_at=datetime.now(),
#                 meeting_point_origin=(avg_origin_lat, avg_origin_lng),
#                 meeting_point_destination=(avg_dest_lat, avg_dest_lng),
#                 status="complete" if len(cluster_users) == 3 else "forming",
#             )
#             clusters.append(cluster.to_dict())

#     return clusters


# # Provided user data
# initial_users = [
#     {
#         "user_id": 1,
#         "origin_lat": 35.803498407040756,
#         "origin_lng": 51.43324473388338,
#         "destination_lat": 35.653855284720954,
#         "destination_lng": 51.38279103608706,
#         "stored_at": "2025-08-27T10:33:39.970122+00:00",
#     },
#     {
#         "user_id": 2,
#         "origin_lat": 35.80122786644687,
#         "origin_lng": 51.43202747906456,
#         "destination_lat": 35.65630422182128,
#         "destination_lng": 51.375871636663746,
#         "stored_at": "2025-08-27T10:33:39.970158+00:00",
#     },
#     {
#         "user_id": 3,
#         "origin_lat": 35.78870250925978,
#         "origin_lng": 51.42791889369835,
#         "destination_lat": 35.65485209103056,
#         "destination_lng": 51.38742627752059,
#         "stored_at": "2025-08-27T10:33:42.059766+00:00",
#     },
#     {
#         "user_id": 4,
#         "origin_lat": 35.785107652011035,
#         "origin_lng": 51.425904384505145,
#         "destination_lat": 35.65048141674927,
#         "destination_lng": 51.37341843733627,
#         "stored_at": "2025-08-27T10:33:42.238852+00:00",
#     },
#     {
#         "user_id": 5,
#         "origin_lat": 35.80344404223181,
#         "origin_lng": 51.437242779087384,
#         "destination_lat": 35.651015447582274,
#         "destination_lng": 51.37336066429711,
#         "stored_at": "2025-08-27T10:33:42.454372+00:00",
#     },
#     {
#         "user_id": 6,
#         "origin_lat": 35.709748621035146,
#         "origin_lng": 51.357650756835945,
#         "destination_lat": 35.788302546115176,
#         "destination_lng": 51.36314392089844,
#         "stored_at": "2025-08-27T10:33:49.539253+00:00",
#     }
# ]

# # Generate clusters
# initial_clusters = create_clusters(initial_users)

main_layout = dbc.Container(
    [
        # Title
        dbc.Row(
            dbc.Col(
                html.H1("User Locations Map", className="text-center text-info mb-4"),
                width=12,
            )
        ),
        dbc.Row(
            [
                # --- Left Column: Controls + Stats ---
                dbc.Col(
                    [
                        # Controls
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Simulation Controls",
                                        className="card-title text-warning mb-3",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="bi bi-person-plus me-2"),
                                            "Add Random User",
                                        ],
                                        id="add-random-user-btn",
                                        n_clicks=0,
                                        color="primary",
                                        className="mb-3 w-100",
                                    ),
                                    html.Label(
                                        "Click Mode:", className="text-light fw-bold"
                                    ),
                                    dcc.RadioItems(
                                        id="click-mode",
                                        options=[
                                            {"label": "Set Origin", "value": "origin"},
                                            {
                                                "label": "Set Destination",
                                                "value": "destination",
                                            },
                                        ],
                                        value="origin",
                                        labelStyle={
                                            "display": "block",
                                            "marginBottom": "5px",
                                        },
                                        className="mb-3",
                                    ),
                                    dbc.Button(
                                        [
                                            html.I(className="bi bi-check-circle me-2"),
                                            "Add Selected User",
                                        ],
                                        id="add-selected-user-btn",
                                        n_clicks=0,
                                        color="success",
                                        className="w-100",
                                    ),
                                ]
                            ),
                            className="mb-4 shadow-sm",
                            style={"borderRadius": "12px"},
                        ),
                        # Stats
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H4(
                                        "Statistics",
                                        className="card-title text-success mb-3",
                                    ),
                                    html.Div(
                                        id="stats-container",
                                        children="Statistics will appear here.",
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                            style={"borderRadius": "12px"},
                        ),
                    ],
                    md=3,  # Narrow column
                ),
                # --- Right Column: Map ---
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Map", className="card-title text-info mb-3"),
                                html.Div(
                                    id="map-container",
                                    children=map_handler.map,
                                    style={
                                        "height": "650px",
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                    },
                                ),
                            ]
                        ),
                        className="shadow-sm",
                        style={"borderRadius": "12px"},
                    ),
                    md=9,  # Wide column for the map
                ),
            ],
            className="g-4",  # nice spacing between cols
        ),
        dcc.Interval(
            id="cluster-update",
            interval=5000,  # in milliseconds (1000 ms = 1 second)
            n_intervals=0,
        ),
        # start at zero)
    ],
    fluid=True,
)

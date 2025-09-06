import dash_leaflet as dl
from dash import html
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from app.service.models import ClusterGroup, UserLocation


class MapHandler:
    def __init__(self, map_id: str = "main-map"):
        self.map_id = map_id
        self.map_center = (35.6892, 51.3890)
        self.zoom = 12
        self.cluster_colors = ["green", "red", "orange", "yellow"]

        self.map = (
            dl.Map(
                [
                    dl.TileLayer(),
                    dl.LayersControl(
                        [
                            dl.Overlay(
                                dl.LayerGroup(id="users"),
                                checked=True,
                                name="User Markers and lines",
                            ),
                            dl.Overlay(
                                dl.LayerGroup(id="clusters"),
                                checked=False,
                                name="Clusters & Meeting Points",
                            ),
                            dl.Overlay(
                                dl.LayerGroup(id="temp-markers"),
                                checked=True,
                                name="Other Components",
                            ),
                        ],
                        id="layer-control",
                    ),
                ],
                id=map_id,
                zoom=12,
                center=self.map_center,
                style={"width": "100%", "height": "100%"},
            ),
        )

    def create_user_marker(
        self, user: UserLocation, is_origin: bool, cluster_color: Optional[str] = None
    ) -> dl.Marker:
        coords = user.origin_coords if is_origin else user.destination_coords
        marker_type = "Origin" if is_origin else "Destination"
        color = cluster_color if cluster_color else "blue"

        return dl.Marker(
            position=coords,
            children=[
                dl.Tooltip(
                    f"User {user.user_id} ({marker_type})"
                    + (f" - Cluster {cluster_color}" if cluster_color else "")
                ),
                dl.Popup(
                    [
                        html.H4(f"User {user.user_id}"),
                        html.P(f"{marker_type}: {coords}"),
                        html.P(f"Stored: {user.stored_at}"),
                        html.P(
                            f"Cluster: {cluster_color if cluster_color else 'None'}"
                        ),
                    ]
                ),
            ],
            icon=dict(
                iconUrl=f"https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-{color}.png",
                iconSize=[25, 41],
                iconAnchor=[12, 41],
                popupAnchor=[1, -34],
            ),
        )

    def create_user_line(self, user: UserLocation) -> dl.Polyline:
        return dl.Polyline(
            positions=[user.origin_coords, user.destination_coords],
            color="gray",
            weight=3,
            children=dl.Tooltip(f"User {user.user_id} Path"),
        )

    def create_cluster_marker(
        self, cluster: ClusterGroup, is_origin: bool, color: str
    ) -> dl.Marker:
        if is_origin and cluster.meeting_point_origin is None:
            return None
        if not is_origin and cluster.meeting_point_destination is None:
            return None

        coords = (
            cluster.meeting_point_origin
            if is_origin
            else cluster.meeting_point_destination
        )
        marker_type = (
            "Meeting Point Origin" if is_origin else "Meeting Point Destination"
        )
        user_ids = ", ".join(str(uid) for uid in cluster.get_user_ids())

        # Use colored star icons directly (no CSS)
        icon_url = "https://raw.githubusercontent.com/iconic/open-iconic/master/png/star-8x.png"

        return dl.Marker(
            position=coords,
            children=[
                dl.Tooltip(f"Cluster {cluster.group_id} ({marker_type})"),
                dl.Popup(
                    [
                        html.H4(f"Cluster {cluster.group_id}"),
                        html.P(f"{marker_type}: {coords}"),
                        html.P(f"Users: {user_ids}"),
                        html.P(f"Status: {cluster.status}"),
                        html.P(f"Created: {cluster.created_at}"),
                    ]
                ),
            ],
            icon=dict(
                iconUrl=icon_url,
                iconSize=[25, 41],
                iconAnchor=[12, 41],
                popupAnchor=[1, -34],
            ),
        )

    def create_cluster_line(self, cluster: ClusterGroup, color: str) -> dl.Polyline:
        if (
            cluster.meeting_point_origin is None
            or cluster.meeting_point_destination is None
        ):
            return None
        return dl.Polyline(
            positions=[cluster.meeting_point_origin, cluster.meeting_point_destination],
            color=color,
            weight=5,
            children=dl.Tooltip(f"Cluster {cluster.group_id} Path"),
        )

    def create_user_to_cluster_lines(
        self, cluster: ClusterGroup, color: str
    ) -> List[dl.Polyline]:
        lines = []
        for user in cluster.users:
            if cluster.meeting_point_origin:
                lines.append(
                    dl.Polyline(
                        positions=[user.origin_coords, cluster.meeting_point_origin],
                        color=color,
                        weight=2,
                        dashArray="5, 10",
                        children=dl.Tooltip(f"User {user.user_id} to Cluster Origin"),
                    )
                )
            if cluster.meeting_point_destination:
                lines.append(
                    dl.Polyline(
                        positions=[
                            user.destination_coords,
                            cluster.meeting_point_destination,
                        ],
                        color=color,
                        weight=2,
                        dashArray="5, 10",
                        children=dl.Tooltip(
                            f"User {user.user_id} to Cluster Destination"
                        ),
                    )
                )
        return lines

    def create_users_and_clusters_layer(
        self,
        users: List[Dict] = None,
        clusters: List[Dict] = None,
    ) -> Tuple[List, List]:
        user_to_cluster = {}
        cluster_layers = []
        user_marker_group = []
        user_line_group = []

        if clusters:
            for idx, cluster_dict in enumerate(clusters):
                cluster = ClusterGroup(
                    group_id=cluster_dict["group_id"],
                    users=[UserLocation.from_dict(u) for u in cluster_dict["users"]],
                    created_at=datetime.fromisoformat(cluster_dict["created_at"]),
                    meeting_point_origin=cluster_dict["meeting_point_origin"],
                    meeting_point_destination=cluster_dict["meeting_point_destination"],
                    status=cluster_dict["status"],
                )
                color = self.cluster_colors[idx % len(self.cluster_colors)]
                for user in cluster.users:
                    user_to_cluster[user.user_id] = color

                cluster_components = [
                    self.create_cluster_marker(cluster, is_origin=True, color=color),
                    self.create_cluster_marker(cluster, is_origin=False, color=color),
                    self.create_cluster_line(cluster, color=color),
                ]
                cluster_components.extend(
                    self.create_user_to_cluster_lines(cluster, color)
                )
                cluster_layers.extend([c for c in cluster_components if c is not None])

        if users:
            for user in users:
                user_obj = UserLocation.from_dict(user)
                cluster_color = user_to_cluster.get(user_obj.user_id)
                user_marker_group.extend(
                    [
                        self.create_user_marker(
                            user_obj, is_origin=True, cluster_color=cluster_color
                        ),
                        self.create_user_marker(
                            user_obj, is_origin=False, cluster_color=cluster_color
                        ),
                    ]
                )
                user_line_group.append(self.create_user_line(user_obj))

        cluster_layers += user_marker_group

        user_layers = user_line_group + user_marker_group

        return user_layers, cluster_layers


map_handler = MapHandler()

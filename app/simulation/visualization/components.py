import dash_leaflet as dl
from typing import List, Dict, Optional, Tuple


from app.service.models import ClusterGroup, UserLocation
from app.service.service import get_clustering_service

from app.service.inter_city_matching.models import InterCityUserLocation
from app.service.inter_city_matching.matching_service import (
    get_inter_city_clustering_service,
)


class MapHandler:
    def __init__(self, prefix: str):
        self.prefix = prefix
        self.map_center = (35.6892, 51.3890)
        self.zoom = 12
        self.cluster_colors = ["green", "red", "orange", "yellow"]

        self.map = dl.Map(
            [
                dl.TileLayer(),
                dl.LayersControl(
                    [
                        dl.Overlay(
                            dl.LayerGroup(id={"type": "users", "prefix": prefix}),
                            checked=True,
                            name="User Markers and lines",
                        ),
                        dl.Overlay(
                            dl.LayerGroup(id={"type": "clusters", "prefix": prefix}),
                            checked=False,
                            name="Clusters & Meeting Points",
                        ),
                        dl.Overlay(
                            dl.LayerGroup(
                                id={"type": "temp-markers", "prefix": prefix}
                            ),
                            checked=True,
                            name="Other Components",
                        ),
                    ],
                    id={"type": "layer-control", "prefix": prefix},
                ),
            ],
            id={"type": "main-map", "prefix": prefix},
            zoom=self.zoom,
            center=self.map_center,
            style={"width": "100%", "height": "100%"},
            preferCanvas=True,
        )

        self.intra_clustering_service = get_clustering_service()
        self.inter_clustering_service = get_inter_city_clustering_service()

    def create_user_marker(
        self, user: UserLocation, is_origin: bool, cluster_color: Optional[str] = None
    ) -> dl.Marker:
        coords = user.origin_coords if is_origin else user.destination_coords
        marker_type = "Origin" if is_origin else "Destination"
        color = cluster_color if cluster_color else "blue"

        return dl.Marker(
            position=coords,
            children=[
                dl.Tooltip(f"User {user.user_id} ({marker_type}) - {coords}"),
            ],
            icon=dict(
                iconUrl=f"https://raw.githubusercontent.com/pointhi/leaflet-color-markers/master/img/marker-icon-2x-{color}.png",
                iconSize=[25, 41],
                iconAnchor=[12, 41],
                popupAnchor=[1, -34],
            ),
        )

    def create_cluster_marker(
        self, cluster: ClusterGroup, is_origin: bool
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
                dl.Tooltip(
                    f"Cluster {cluster.group_id} ({marker_type}) - {coords} \n Users: {user_ids}"
                ),
            ],
            icon=dict(
                iconUrl=icon_url,
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
            weight=3,
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
                    )
                )

        return [dl.FeatureGroup(lines)]

    def generate_user_to_cluster(self, clusters: List[ClusterGroup] = None) -> Dict:
        user_to_cluster = {}
        for idx, cluster in enumerate(clusters):
            color = self.cluster_colors[idx % len(self.cluster_colors)]
            for user in cluster.users:
                user_to_cluster[user.user_id] = color
        return user_to_cluster

    def create_clusters_layer(
        self,
        clusters: List[ClusterGroup] = None,
    ) -> Tuple[List, List]:
        cluster_layers = []

        if clusters:
            for idx, cluster in enumerate(clusters):
                color = self.cluster_colors[idx % len(self.cluster_colors)]
                cluster_components = [
                    self.create_cluster_marker(cluster, is_origin=True),
                    self.create_cluster_marker(cluster, is_origin=False),
                    self.create_cluster_line(cluster, color=color),
                ]
                cluster_components.extend(
                    self.create_user_to_cluster_lines(cluster, color=color)
                )
                cluster_layers.extend([c for c in cluster_components if c is not None])

        return cluster_layers

    def create_users_layer(
        self,
        users: List[UserLocation] = None,
        clusters: List[ClusterGroup] = None,
    ) -> Tuple[List, List]:
        user_marker_group = []
        user_line_group = []
        user_to_cluster = self.generate_user_to_cluster(clusters)
        if users:
            for user in users:
                cluster_color = user_to_cluster.get(user.user_id)
                user_marker_group.extend(
                    [
                        self.create_user_marker(
                            user, is_origin=True, cluster_color=cluster_color
                        ),
                        self.create_user_marker(
                            user, is_origin=False, cluster_color=cluster_color
                        ),
                    ]
                )
                user_line_group.append(self.create_user_line(user))
        user_layers = [
            dl.FeatureGroup(children=user_line_group),
            dl.FeatureGroup(children=user_marker_group),
        ]

        return user_layers

    def create_route_polyline(
        self,
        route_coords: List[Tuple[float, float]],
        route_id: str,
    ) -> dl.Polyline:
        """
        Create a polyline to display a route on the map.
        """
        if not route_coords or len(route_coords) < 2:
            return None

        return dl.Polyline(
            positions=route_coords,
            color="blue",
            weight=3,
            opacity=0.8,
            id={"type": "route", "route_id": route_id, "prefix": self.prefix},
            children=[
                dl.Tooltip(
                    f"Route {route_id}",
                    permanent=False,
                    direction="auto",
                    sticky=True,
                    opacity=0.9,
                )
            ],
        )

    def create_users_layer_inter_city(
        self,
        users: List[InterCityUserLocation] = None,
    ) -> Tuple[List, List]:
        user_layers = []
        if users:
            for idx, user in enumerate(users):
                user_layers.append(
                    
                    self.create_route_polyline(
                        route_coords=self.inter_clustering_service.fetch_route_for_user(user=user),
                        route_id=str(idx)
                    )
                )

        return user_layers

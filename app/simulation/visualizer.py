import folium


class RouteVisualizer:
    def __init__(self, route_data):
        """
        Initialize RouteVisualizer with route data.

        :param route_data: List of lists containing 
               [origin_lat, origin_lng, dest_lat, dest_lng]
        """
        self.route_data = route_data
        self.map = None

    def _calculate_center(self):
        """Calculate map center point from all coordinates"""
        all_lats = []
        all_lngs = []
        for route in self.route_data:
            all_lats.extend([route[0], route[2]])  # Origin and dest lats
            all_lngs.extend([route[1], route[3]])  # Origin and dest lngs
        return [sum(all_lats)/len(all_lats), sum(all_lngs)/len(all_lngs)]

    def _generate_popup_html(self, pair_id, is_origin):
        """Generate HTML content for marker popups"""
        location_type = "Origin" if is_origin else "Destination"
        return f"""
        <div style="font-family: Arial; text-align: center;">
            <b>Pair ID: {pair_id}</b><br>
            <b>Type: {location_type}</b><br>
            <button style="margin-top: 8px; padding: 6px 12px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer;"
                    onclick="drawLine({pair_id})">
                Show Connection
            </button>
        </div>
        """

    def visualize(self, output_file="route_map.html"):
        """Generate interactive map visualization"""
        # Create map with calculated center
        self.map = folium.Map(
            location=self._calculate_center(),
            zoom_start=12,
            tiles="cartodbpositron"
        )

        # Create a feature group for connecting lines
        self.lines_group = folium.FeatureGroup(name="Connecting Lines")
        self.map.add_child(self.lines_group)

        # Prepare JavaScript for line drawing functionality
        line_coords_js = []
        for i, route in enumerate(self.route_data):
            line_coords_js.append(
                f"[ [{route[0]}, {route[1]}], [{route[2]}, {route[3]}] ]")

        # Get the unique group name for JavaScript reference
        group_name = self.lines_group.get_name().replace(
            ' ', '_') + '_' + str(id(self.lines_group))

        js_code = f"""
        <script>
        // Store route coordinates
        const routeCoords = [
            {','.join(line_coords_js)}
        ];
        
        // Function to draw connection lines
        function drawLine(pairId) {{
            // Access the feature group
            const linesGroup = {group_name};
            
            // Clear existing lines
            linesGroup.clearLayers();
            
            // Create new polyline
            const coords = routeCoords[pairId];
            const line = L.polyline(
                coords,
                {{
                    color: '#FF5733',
                    weight: 4,
                    opacity: 0.7,
                    smoothFactor: 1
                }}
            );
            
            // Add to feature group
            line.addTo(linesGroup);
        }}
        </script>
        """

        # Add JavaScript to map
        self.map.get_root().html.add_child(folium.Element(js_code))

        # Create markers for each point
        for i, route in enumerate(self.route_data):
            # Origin marker
            origin_popup = folium.Popup(
                html=self._generate_popup_html(i, True),
                max_width=250
            )
            folium.Marker(
                location=[route[0], route[1]],
                popup=origin_popup,
                icon=folium.Icon(color='green', icon='circle', prefix='fa'),
                tooltip=f"Origin {i}"
            ).add_to(self.map)

            # Destination marker
            dest_popup = folium.Popup(
                html=self._generate_popup_html(i, False),
                max_width=250
            )
            folium.Marker(
                location=[route[2], route[3]],
                popup=dest_popup,
                icon=folium.Icon(color='red', icon='flag', prefix='fa'),
                tooltip=f"Destination {i}"
            ).add_to(self.map)

        # Add layer control to toggle lines visibility
        folium.LayerControl().add_to(self.map)

        self.map.save(output_file)
        return self.map


# Usage example
if __name__ == "__main__":
    data = [
        [35.8865478, 50.7093883, 35.9885159, 50.7392578],
        [35.8888986, 50.7112983, 35.9913181, 50.740812],
        [35.8865852, 50.7065667, 35.9885159, 50.7392578],
        [35.884459, 50.7117233, 35.9890637, 50.73844],
        [35.8852425, 50.7108559, 35.9928111, 50.7416271],
        [35.8863276, 50.7083118, 35.9902971, 50.7386223]
    ]

    visualizer = RouteVisualizer(data)
    visualizer.visualize("routes_map.html")
    print("Map saved as 'routes_map.html'")

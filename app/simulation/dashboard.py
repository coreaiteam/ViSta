# dashboard.py - Main Dash dashboard class
import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import threading
import time
from datetime import datetime
from queue import Queue
import json

from .service_manager import ServiceManager


class ClusteringDashboard:
    """Dash-based dashboard for clustering service"""

    def __init__(self, host='127.0.0.1', port=8050, debug=True):
        self.host = host
        self.port = port
        self.debug = debug

        # Initialize service manager
        self.service_manager = ServiceManager()

        # Data storage
        self.results_queue = Queue()
        self.submitted_locations = []
        self.results_data = []

        # Create Dash app
        self.app = dash.Dash(__name__, title="Clustering Service Dashboard")
        self._setup_layout()
        self._setup_callbacks()

        # Background thread for handling results
        self.result_thread = None
        self.running = False

    def _setup_layout(self):
        """Setup the dashboard layout"""
        self.app.layout = html.Div([
            dcc.Store(id='locations-store', data=[]),
            dcc.Store(id='results-store', data=[]),
            dcc.Interval(id='interval-component',
                         interval=1000, n_intervals=0),

            html.Div([
                html.H1("🗺️ Clustering Service Dashboard", className="header"),

                # Input panel
                html.Div([
                    html.H3("📍 Submit Location"),
                    html.Div([
                        html.Div([
                            html.Label("Origin Latitude:"),
                            dcc.Input(id='origin-lat', type='number',
                                      value=35.7219, step=0.0001)
                        ], className="input-group"),

                        html.Div([
                            html.Label("Origin Longitude:"),
                            dcc.Input(id='origin-lng', type='number',
                                      value=51.3347, step=0.0001)
                        ], className="input-group"),

                        html.Div([
                            html.Label("Destination Latitude:"),
                            dcc.Input(id='dest-lat', type='number',
                                      value=35.7319, step=0.0001)
                        ], className="input-group"),

                        html.Div([
                            html.Label("Destination Longitude:"),
                            dcc.Input(id='dest-lng', type='number',
                                      value=51.3447, step=0.0001)
                        ], className="input-group"),
                    ], className="input-row"),

                    html.Div([
                        html.Button("Submit Location", id="submit-btn",
                                    n_clicks=0, className="submit-button"),
                        html.Button("Clear All", id="clear-btn",
                                    n_clicks=0, className="clear-button"),
                    ], className="button-row"),

                    html.Div(id="status-message", className="status-message")
                ], className="input-panel"),

                # Statistics panel
                html.Div([
                    html.H3("📊 Statistics"),
                    html.Div([
                        html.Div([
                            html.H2(id="locations-count", children="0"),
                            html.P("Locations Submitted")
                        ], className="stat-box"),

                        html.Div([
                            html.H2(id="results-count", children="0"),
                            html.P("Results Received")
                        ], className="stat-box"),

                        html.Div([
                            html.H2(id="service-status", children="●"),
                            html.P("Service Status")
                        ], className="stat-box"),
                    ], className="stats-row")
                ], className="stats-panel"),

                # Map
                html.Div([
                    html.H3("🗺️ Map View"),
                    dcc.Graph(id="map-graph", style={'height': '400px'})
                ], className="map-panel"),

                # Results
                html.Div([
                    html.H3("📡 Live Results"),
                    html.Div(id="results-display", className="results-display")
                ], className="results-panel"),

            ], className="container")
        ])

        # Add CSS styling
        self.app.index_string = '''
        <!DOCTYPE html>
        <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; }
                .header { color: #333; text-align: center; margin-bottom: 30px; }
                
                .input-panel, .stats-panel, .map-panel, .results-panel {
                    background: white; padding: 20px; margin-bottom: 20px; 
                    border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                
                .input-row { display: flex; gap: 15px; margin-bottom: 15px; }
                .input-group { flex: 1; }
                .input-group label { display: block; margin-bottom: 5px; font-weight: bold; color: #555; }
                
                .button-row { display: flex; gap: 10px; margin-top: 15px; }
                .submit-button { background: #007bff; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                .clear-button { background: #dc3545; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; }
                .submit-button:hover { background: #0056b3; }
                .clear-button:hover { background: #c82333; }
                
                .stats-row { display: flex; gap: 20px; justify-content: center; }
                .stat-box { text-align: center; padding: 15px; background: #f8f9fa; border-radius: 8px; min-width: 120px; }
                .stat-box h2 { margin: 0; font-size: 2em; color: #007bff; }
                .stat-box p { margin: 5px 0 0 0; color: #666; font-size: 0.9em; }
                
                .status-message { margin-top: 15px; padding: 10px; border-radius: 4px; }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
                
                .results-display { max-height: 300px; overflow-y: auto; background: #f8f9fa; padding: 15px; border-radius: 4px; }
                .result-item { margin-bottom: 10px; padding: 10px; background: white; border-radius: 4px; border-left: 4px solid #007bff; }
                .result-time { font-size: 0.8em; color: #666; margin-bottom: 5px; }
                .result-data { font-family: monospace; font-size: 0.9em; }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
        </html>
        '''

    def _setup_callbacks(self):
        """Setup Dash callbacks"""

        @self.app.callback(
            [Output('locations-store', 'data'),
             Output('status-message', 'children'),
             Output('status-message', 'className')],
            Input('submit-btn', 'n_clicks'),
            [State('origin-lat', 'value'),
             State('origin-lng', 'value'),
             State('dest-lat', 'value'),
             State('dest-lng', 'value'),
             State('locations-store', 'data')]
        )
        def submit_location(n_clicks, origin_lat, origin_lng, dest_lat, dest_lng, current_data):
            if n_clicks == 0:
                return current_data, "", ""

            try:
                # Validate inputs
                if None in [origin_lat, origin_lng, dest_lat, dest_lng]:
                    return current_data, "❌ Please fill all coordinates", "status-message error"

                # Submit to service
                request_id = self.service_manager.submit_location(
                    origin_lat, origin_lng, dest_lat, dest_lng)

                # Add to stored data
                location_data = {
                    'id': request_id,
                    'origin_lat': origin_lat,
                    'origin_lng': origin_lng,
                    'dest_lat': dest_lat,
                    'dest_lng': dest_lng,
                    'timestamp': datetime.now().isoformat()
                }

                updated_data = current_data + [location_data]

                return updated_data, f"✅ Location submitted: {request_id}", "status-message success"

            except Exception as e:
                return current_data, f"❌ Error: {str(e)}", "status-message error"

        @self.app.callback(
            Output('locations-store', 'data', allow_duplicate=True),
            Input('clear-btn', 'n_clicks'),
            prevent_initial_call=True
        )
        def clear_locations(n_clicks):
            if n_clicks > 0:
                self.results_data.clear()
                return []
            return dash.no_update

        @self.app.callback(
            [Output('map-graph', 'figure'),
             Output('locations-count', 'children'),
             Output('results-count', 'children'),
             Output('service-status', 'children'),
             Output('service-status', 'style'),
             Output('results-display', 'children'),
             Output('results-store', 'data')],
            Input('interval-component', 'n_intervals'),
            State('locations-store', 'data')
        )
        def update_dashboard(n_intervals, locations_data):
            # Update results from queue
            new_results = []
            while not self.results_queue.empty():
                try:
                    result = self.results_queue.get_nowait()
                    self.results_data.append(result)
                    new_results.append(result)
                except:
                    break

            # Create map
            fig = self._create_map(locations_data)

            # Update stats
            locations_count = len(locations_data) if locations_data else 0
            results_count = len(self.results_data)

            service_running = self.service_manager.is_running()
            service_status = "●"
            service_style = {
                'color': '#28a745' if service_running else '#dc3545'}

            # Create results display
            results_display = self._create_results_display()

            return fig, locations_count, results_count, service_status, service_style, results_display, self.results_data

    def _create_map(self, locations_data):
        """Create the map visualization"""
        fig = go.Figure()

        if locations_data:
            # Add origin points
            origins_lat = [loc['origin_lat'] for loc in locations_data]
            origins_lng = [loc['origin_lng'] for loc in locations_data]

            fig.add_trace(go.Scattermapbox(
                lat=origins_lat,
                lon=origins_lng,
                mode='markers',
                marker=dict(size=10, color='green'),
                name='Origins',
                text=[f"Origin {i+1}" for i in range(len(origins_lat))]
            ))

            # Add destination points
            dests_lat = [loc['dest_lat'] for loc in locations_data]
            dests_lng = [loc['dest_lng'] for loc in locations_data]

            fig.add_trace(go.Scattermapbox(
                lat=dests_lat,
                lon=dests_lng,
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Destinations',
                text=[f"Destination {i+1}" for i in range(len(dests_lat))]
            ))

            # Add lines connecting origin to destination
            for loc in locations_data:
                fig.add_trace(go.Scattermapbox(
                    lat=[loc['origin_lat'], loc['dest_lat']],
                    lon=[loc['origin_lng'], loc['dest_lng']],
                    mode='lines',
                    line=dict(width=2, color='blue'),
                    showlegend=False
                ))

            # Center map on data
            center_lat = sum(origins_lat + dests_lat) / \
                len(origins_lat + dests_lat)
            center_lng = sum(origins_lng + dests_lng) / \
                len(origins_lng + dests_lng)
        else:
            # Default center (Tehran)
            center_lat, center_lng = 35.7219, 51.3347

        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(lat=center_lat, lon=center_lng),
                zoom=10
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=400
        )

        return fig

    def _create_results_display(self):
        """Create results display components"""
        if not self.results_data:
            return html.Div("No results yet...", className="result-item")

        results_components = []
        # Show last 10 results
        for result in reversed(self.results_data[-10:]):
            results_components.append(
                html.Div([
                    html.Div(result.get('timestamp', 'Unknown time'),
                             className="result-time"),
                    html.Div(json.dumps(result.get('data', {}),
                             indent=2), className="result-data")
                ], className="result-item")
            )

        return results_components

    def start(self):
        """Start the dashboard and service"""
        print("🚀 Starting clustering service...")
        self.service_manager.start()
        self.service_manager.set_result_callback(self._handle_result)

        print("🚀 Starting dashboard...")
        self.running = True

        # Start result handling thread
        self.result_thread = threading.Thread(
            target=self._result_handler, daemon=True)
        self.result_thread.start()

        print(f"🌐 Dashboard available at: http://{self.host}:{self.port}")

        try:
            self.app.run(
                host=self.host, port=self.port, debug=self.debug)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        """Stop the dashboard and service"""
        print("\n🛑 Stopping dashboard...")
        self.running = False
        self.service_manager.stop()

    def _handle_result(self, result):
        """Handle results from the service"""
        result_data = {
            'timestamp': datetime.now().isoformat(),
            'data': result
        }
        self.results_queue.put(result_data)

    def _result_handler(self):
        """Background thread to handle results"""
        while self.running:
            time.sleep(0.1)  # Small delay to prevent high CPU usage

# import dash_bootstrap_components as dbc
# from dash import html, dcc  # noqa: F811
# from .components import map_handler


# inter_city_layout = (
#     dbc.Row(
#         [
#             dbc.Col(
#                 dbc.Card(
#                     dbc.CardBody(
#                         [
#                             html.H4(
#                                 "Inter-City Clustering Map",
#                                 className="card-title text-primary mb-3",
#                             ),
#                             html.Div(
#                                 id="intercity-map-container",
#                                 style={
#                                     "height": "calc(100vh - 150px)",
#                                     "borderRadius": "12px",
#                                     "overflow": "hidden",
#                                 },
#                             ),
#                         ]
#                     ),
#                     className="shadow-sm h-100",
#                     style={"borderRadius": "12px"},
#                 ),
#                 width=12,
#             )
#         ],
#         style={"minHeight": "85vh"},
#     ),
# )


# intra_city_layout = (
#     dbc.Row(
#         [
#             # Left column (controls + stats)
#             dbc.Col(
#                 [
#                     # Controls Accordion
#                     dbc.Accordion(
#                         [
#                             dbc.AccordionItem(
#                                 [
#                                     dbc.Button(
#                                         [
#                                             html.I(className="bi bi-person-plus me-2"),
#                                             "Add Random User",
#                                         ],
#                                         id="add-random-user-btn",
#                                         n_clicks=0,
#                                         color="primary",
#                                         className="mb-3 w-100",
#                                     ),
#                                     dbc.InputGroup(
#                                         [
#                                             dbc.Input(
#                                                 id="bulk-user-count",
#                                                 type="number",
#                                                 min=1,
#                                                 step=1,
#                                                 debounce=True,
#                                                 placeholder="Number of users",
#                                             ),
#                                             dbc.Button(
#                                                 [
#                                                     html.I(
#                                                         className="bi bi-people-fill me-2"
#                                                     ),
#                                                     "Add Multiple Users",
#                                                 ],
#                                                 id="add-multiple-users-btn",
#                                                 n_clicks=0,
#                                                 color="info",
#                                             ),
#                                         ],
#                                         className="mb-3",
#                                     ),
#                                     html.Div(
#                                         id="bulk-user-info",
#                                         className="text-muted small mb-3",
#                                     ),
#                                     html.Label(
#                                         "Click Mode:", className="text-light fw-bold"
#                                     ),
#                                     dcc.RadioItems(
#                                         id="click-mode",
#                                         options=[
#                                             {"label": "Set Origin", "value": "origin"},
#                                             {
#                                                 "label": "Set Destination",
#                                                 "value": "destination",
#                                             },
#                                         ],
#                                         value="origin",
#                                         labelStyle={
#                                             "display": "block",
#                                             "marginBottom": "5px",
#                                         },
#                                         className="mb-3",
#                                     ),
#                                     dbc.Button(
#                                         [
#                                             html.I(className="bi bi-x-circle me-2"),
#                                             "Clear Markers",
#                                         ],
#                                         id="clear-markers-btn",
#                                         n_clicks=0,
#                                         color="danger",
#                                         className="w-100 mb-2",
#                                     ),
#                                     dbc.Button(
#                                         [
#                                             html.I(className="bi bi-check-circle me-2"),
#                                             "Add Selected User",
#                                         ],
#                                         id="add-selected-user-btn",
#                                         n_clicks=0,
#                                         color="success",
#                                         className="w-100 mb-3",
#                                     ),
#                                     dbc.InputGroup(
#                                         [
#                                             dbc.Input(
#                                                 id="remove-user-id",
#                                                 type="number",
#                                                 placeholder="Enter User ID",
#                                                 min=1,
#                                                 debounce=True,
#                                             ),
#                                             dbc.Button(
#                                                 [
#                                                     html.I(
#                                                         className="bi bi-trash me-2"
#                                                     ),
#                                                     "Remove User",
#                                                 ],
#                                                 id="remove-user-btn",
#                                                 n_clicks=0,
#                                                 color="warning",
#                                             ),
#                                         ],
#                                     ),
#                                 ],
#                                 title="Simulation Controls ⚙️",
#                                 className="mb-4",
#                             ),
#                         ],
#                         start_collapsed=False,
#                         className="shadow-sm",
#                         style={"borderRadius": "12px"},
#                     ),
#                     # Stats Accordion
#                     dbc.Accordion(
#                         [
#                             dbc.AccordionItem(
#                                 [
#                                     html.Div(
#                                         [
#                                             html.H4(
#                                                 "Live Statistics",
#                                                 className="text-success fw-bold mb-3",
#                                             ),
#                                             html.Div(
#                                                 id="stats-container",
#                                                 children=[
#                                                     html.Ul(
#                                                         [
#                                                             html.Li(
#                                                                 "Total Users: Loading..."
#                                                             ),
#                                                             html.Li(
#                                                                 "Active Matches: Loading..."
#                                                             ),
#                                                             html.Li(
#                                                                 "Clusters Formed: Loading..."
#                                                             ),
#                                                         ],
#                                                         className="list-group list-group-flush",
#                                                     )
#                                                 ],
#                                             ),
#                                         ],
#                                         className="p-2",
#                                     )
#                                 ],
#                                 title="Statistics Overview 📊",
#                             ),
#                         ],
#                         start_collapsed=False,
#                         className="shadow-sm mt-4",
#                         style={"borderRadius": "12px"},
#                     ),
#                 ],
#                 xs=12,
#                 md=3,
#             ),


#             # Right column (map)
#             dbc.Col(
#                 dbc.Card(
#                     dbc.CardBody(
#                         [
#                             html.H4("Map", className="card-title text-info mb-3 text-center"),
#                             html.Div(
#                                 id="map-container",
#                                 children=map_handler.map,
#                                 style={
#                                     "height": "calc(100vh - 150px)",
#                                     "borderRadius": "12px",
#                                     "overflow": "hidden",
#                                 },
#                             ),
#                         ],
#                         className="h-100 p-0",
#                     ),
#                     className="shadow-sm h-100",
#                     style={"borderRadius": "12px"},
#                 ),
#                 xs=12,
#                 md=9,
#                 className="mt-4 mt-md-0",
#             ),


#         ],
#         className="g-4",
#         style={"minHeight": "85vh"},
#     ),
# )


# main_layout = dbc.Container(
#     [
#         # Tabs
#         dbc.Tabs(
#             [
#                 # --- Tab 1: Intra-city clustering ---
#                 dbc.Tab(
#                     label="Intra-City Clustering",
#                     tab_id="intra-city",
#                     children=dbc.Container(
#                         intra_city_layout,
#                         fluid=True,
#                         className="p-3 bg-dark text-light rounded shadow-sm",
#                         style={"minHeight": "80vh"},
#                     ),
#                 ),
#                 # --- Tab 2: Inter-city clustering ---
#                 dbc.Tab(
#                     label="Inter-City Clustering",
#                     tab_id="inter-city",
#                     children=dbc.Container(
#                         inter_city_layout,
#                         fluid=True,
#                         className="p-3 bg-dark text-light rounded shadow-sm",
#                         style={"minHeight": "80vh"},
#                     ),
#                 ),
#             ],
#             id="clustering-tabs",
#             active_tab="intra-city",
#             className="mb-3",  # space between tab bar and content
#         ),

#         # Tooltips
#         dbc.Tooltip("Add a single random user to the map", target="add-random-user-btn"),
#         dbc.Tooltip("Remove a specific user by ID", target="remove-user-btn"),

#         # Intervals
#         dcc.Interval(id="users-refresh-interval", interval=5000, n_intervals=0),
#         dcc.Interval(id="stats-refresh-interval", interval=2000, n_intervals=0),
#     ],
#     fluid=True,
#     style={"height": "100vh", "padding": "20px"},
# )


import dash_bootstrap_components as dbc
from dash import html, dcc  # noqa: F811
from .components import MapHandler


# -------------------------------
# Controls Section
# -------------------------------
def make_controls_section(prefix: str):
    return dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    dbc.Button(
                        [
                            html.I(className="bi bi-person-plus me-2"),
                            "Add Random User",
                        ],
                        id={"type": "add-random-user-btn", "prefix": prefix},
                        n_clicks=0,
                        color="primary",
                        className="mb-3 w-100",
                    ),
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id={"type": "bulk-user-count", "prefix": prefix},
                                type="number",
                                min=1,
                                step=1,
                                debounce=True,
                                placeholder="Number of users",
                            ),
                            dbc.Button(
                                [
                                    html.I(className="bi bi-people-fill me-2"),
                                    "Add Multiple Users",
                                ],
                                id={"type": "add-multiple-users-btn", "prefix": prefix},
                                n_clicks=0,
                                color="info",
                            ),
                        ],
                        className="mb-3",
                    ),
                    html.Div(
                        id={"type": "bulk-user-info", "prefix": prefix},
                        className="text-muted small mb-3",
                    ),
                    html.Label("Click Mode:", className="text-light fw-bold"),
                    dcc.RadioItems(
                        id={"type": "click-mode", "prefix": prefix},
                        options=[
                            {"label": "Set Origin", "value": "origin"},
                            {"label": "Set Destination", "value": "destination"},
                        ],
                        value="origin",
                        labelStyle={"display": "block", "marginBottom": "5px"},
                        className="mb-3",
                    ),
                    dbc.Button(
                        [
                            html.I(className="bi bi-x-circle me-2"),
                            "Clear Markers",
                        ],
                        id={"type": "clear-markers-btn", "prefix": prefix},
                        n_clicks=0,
                        color="danger",
                        className="w-100 mb-2",
                    ),
                    dbc.Button(
                        [
                            html.I(className="bi bi-check-circle me-2"),
                            "Add Selected User",
                        ],
                        id={"type": "add-selected-user-btn", "prefix": prefix},
                        n_clicks=0,
                        color="success",
                        className="w-100 mb-3",
                    ),
                    dbc.InputGroup(
                        [
                            dbc.Input(
                                id={"type": "remove-user-id", "prefix": prefix},
                                type="number",
                                placeholder="Enter User ID",
                                min=1,
                                debounce=True,
                            ),
                            dbc.Button(
                                [
                                    html.I(className="bi bi-trash me-2"),
                                    "Remove User",
                                ],
                                id={"type": "remove-user-btn", "prefix": prefix},
                                n_clicks=0,
                                color="warning",
                            ),
                        ],
                    ),
                ],
                title="Simulation Controls ⚙️",
                className="mb-4",
            ),
        ],
        start_collapsed=False,
        className="shadow-sm",
        style={"borderRadius": "12px"},
    )


# -------------------------------
# Stats Section
# -------------------------------
def make_stats_section(prefix: str):
    return dbc.Accordion(
        [
            dbc.AccordionItem(
                [
                    html.Div(
                        [
                            html.H4(
                                "Live Statistics",
                                className="text-success fw-bold mb-3",
                            ),
                            html.Div(
                                id={"type": "stats-container", "prefix": prefix},
                                children=[
                                    html.Ul(
                                        [
                                            html.Li("Total Users: Loading..."),
                                            html.Li("Active Matches: Loading..."),
                                            html.Li("Clusters Formed: Loading..."),
                                        ],
                                        className="list-group list-group-flush",
                                    )
                                ],
                            ),
                        ],
                        className="p-2",
                    )
                ],
                title="Statistics Overview 📊",
            ),
        ],
        start_collapsed=False,
        className="shadow-sm mt-4",
        style={"borderRadius": "12px"},
    )


# -------------------------------
# Map Section
# -------------------------------
def make_map_section(prefix: str, map_component=None, title="Map"):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(title, className="card-title text-info mb-3 text-center"),
                html.Div(
                    id={"type": "map-container", "prefix": prefix},
                    children=map_component,
                    style={
                        "height": "calc(100vh - 150px)",
                        "borderRadius": "12px",
                        "overflow": "hidden",
                    },
                ),
                # # لایه‌های کاربران و خوشه‌ها
                # html.Div(id={"type": "users", "prefix": prefix}, style={"display": "none"}),
                # html.Div(id={"type": "clusters", "prefix": prefix}, style={"display": "none"}),
                # html.Div(id={"type": "temp-markers", "prefix": prefix}, style={"display": "none"}),
            ],
            className="h-100 p-0",
        ),
        className="shadow-sm h-100",
        style={"borderRadius": "12px"},
    )


# -------------------------------
# Intra-city Layout
# -------------------------------
intra_map_handler = MapHandler(prefix="intra")
intra_city_layout = dbc.Row(
    [
        dbc.Col(
            [
                make_controls_section("intra"),
                make_stats_section("intra"),
            ],
            xs=12,
            md=3,
        ),
        dbc.Col(
            make_map_section("intra", intra_map_handler.map, "Intra-City Map"),
            xs=12,
            md=9,
            className="mt-4 mt-md-0",
        ),
    ],
    className="g-4",
    style={"minHeight": "85vh"},
)


# -------------------------------
# Inter-city Layout
# -------------------------------
inter_map_handler = MapHandler(prefix="inter")
inter_city_layout = dbc.Row(
    [
        dbc.Col(
            [
                make_controls_section("inter"),
                make_stats_section("inter"),
            ],
            xs=12,
            md=3,
        ),
        dbc.Col(
            make_map_section("inter", inter_map_handler.map, "Inter-City Map"),
            xs=12,
            md=9,
            className="mt-4 mt-md-0",
        ),
    ],
    className="g-4",
    style={"minHeight": "85vh"},
)


# -------------------------------
# Main Layout
# -------------------------------
main_layout = dbc.Container(
    [
        dbc.Tabs(
            [
                dbc.Tab(
                    label="Intra-City Clustering",
                    tab_id="intra-city",
                    children=dbc.Container(
                        intra_city_layout,
                        fluid=True,
                        className="p-3 bg-dark text-light rounded shadow-sm",
                        style={"minHeight": "80vh"},
                    ),
                ),
                dbc.Tab(
                    label="Inter-City Clustering",
                    tab_id="inter-city",
                    children=dbc.Container(
                        inter_city_layout,
                        fluid=True,
                        className="p-3 bg-dark text-light rounded shadow-sm",
                        style={"minHeight": "80vh"},
                    ),
                ),
            ],
            id="clustering-tabs",
            active_tab="intra-city",
            className="mb-3",
        ),
        # Tooltips (الان هم dict شدن)
        dbc.Tooltip(
            "Add a single random user to the map",
            target={"type": "add-random-user-btn", "prefix": "intra"},
        ),
        dbc.Tooltip(
            "Remove a specific user by ID",
            target={"type": "remove-user-btn", "prefix": "intra"},
        ),
        dbc.Tooltip(
            "Add a single random user to the map",
            target={"type": "add-random-user-btn", "prefix": "inter"},
        ),
        dbc.Tooltip(
            "Remove a specific user by ID",
            target={"type": "remove-user-btn", "prefix": "inter"},
        ),
        # Intervals (الان هم prefix دارن)
        dcc.Interval(
            id={"type": "users-refresh-interval", "prefix": "intra"},
            interval=5000,
            n_intervals=0,
        ),
        dcc.Interval(
            id={"type": "stats-refresh-interval", "prefix": "intra"},
            interval=2000,
            n_intervals=0,
        ),
        # dcc.Interval(
        #     id={"type": "users-refresh-interval", "prefix": "inter"},
        #     interval=5000,
        #     n_intervals=0,
        # ),
        # dcc.Interval(
        #     id={"type": "stats-refresh-interval", "prefix": "inter"},
        #     interval=2000,
        #     n_intervals=0,
        # ),
    ],
    fluid=True,
    style={"height": "100vh", "padding": "20px"},
)

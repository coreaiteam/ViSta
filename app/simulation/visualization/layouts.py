import dash_bootstrap_components as dbc

from dash import html, dcc  # noqa: F811

from .components import map_handler


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
                                    dbc.InputGroup(
                                        [
                                            dbc.Input(
                                                id="bulk-user-count",
                                                type="number",
                                                min=1,
                                                step=1,
                                                placeholder="Number of users",
                                            ),
                                            dbc.Button(
                                                [
                                                    html.I(
                                                        className="bi bi-people-fill me-2"
                                                    ),
                                                    "Add Multiple Users",
                                                ],
                                                id="add-multiple-users-btn",
                                                n_clicks=0,
                                                color="info",
                                            ),
                                        ],
                                        className="mb-3",
                                    ),
                                    html.Div(
                                        id="bulk-user-info",
                                        className="text-muted small mb-3",
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
                                            html.I(className="bi bi-x-circle me-2"),
                                            "Clear Markers",
                                        ],
                                        id="clear-markers-btn",
                                        n_clicks=0,
                                        color="danger",
                                        className="w-100",
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
                                    html.Div(
                                        [
                                            dbc.InputGroup(
                                                [
                                                    dbc.Input(
                                                        id="remove-user-id",
                                                        type="number",
                                                        placeholder="Enter User ID",
                                                        min=1,
                                                    ),
                                                    dbc.Button(
                                                        [
                                                            html.I(
                                                                className="bi bi-trash me-2"
                                                            ),
                                                            "Remove User",
                                                        ],
                                                        id="remove-user-btn",
                                                        n_clicks=0,
                                                        color="warning",
                                                    ),
                                                ],
                                                className="mt-3",
                                            )
                                        ]
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
                    md=3,  # narrow left column
                ),
                # --- Right Column: Map (full height) ---
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Map", className="card-title text-info mb-3"),
                                html.Div(
                                    id="map-container",
                                    children=map_handler.map,
                                    style={
                                        "height": "100%",  # take full height
                                        "minHeight": "100%",  # ensure stretch
                                        "borderRadius": "8px",
                                        "overflow": "hidden",
                                    },
                                ),
                            ],
                            className="h-100",  # stretch card body
                        ),
                        className="shadow-sm h-100",  # stretch card
                        style={"borderRadius": "12px"},
                    ),
                    md=9,
                    className="h-100",  # stretch column
                ),
            ],
            className="g-4",
            style={"height": "85vh"},  # full row height (adjust as needed)
        ),
        # Intervals
        dcc.Interval(
            id="users-refresh-interval",
            interval=5 * 1000,
            n_intervals=0,
        ),
        dcc.Interval(
            id="stats-refresh-interval",
            interval=1 * 1000,
            n_intervals=0,
        ),
    ],
    fluid=True,
    style={"height": "100vh"},  # make container full screen height
)

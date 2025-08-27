from .dashboard import ClusteringDashboard

if __name__ == "__main__":
    # Create and run dashboard
    dashboard = ClusteringDashboard(host='127.0.0.1', port=8050, debug=True)
    dashboard.start()

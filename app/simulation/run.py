from .dashboard import ClusteringDashboard

from .visualization.main import app
if __name__ == "__main__":
    # Create and run dashboard
    dashboard = ClusteringDashboard(app=app)
    dashboard.start()
    

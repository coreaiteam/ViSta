# service_manager.py - Simple service manager
from datetime import datetime, timezone
from typing import Callable, Optional

from ..service.service import get_clustering_service
from ..service.models import UserLocation


class ServiceManager:
    """Manages the clustering service"""

    def __init__(self):
        self.service = None
        self.request_count = 0
        self.result_callback: Optional[Callable] = None

    def start(self):
        """Start the clustering service"""
        self.service = get_clustering_service(clustering_interval=5)

        # Add callback handler
        self.service.add_output_handler({
            "type": "callback",
            "callback": self._handle_service_result
        })

        self.service.start()
        print("✅ Clustering service started")

    def stop(self):
        """Stop the service"""
        if self.service:
            self.service.stop()
            print("✅ Clustering service stopped")

    def is_running(self):
        """Check if service is running"""
        return self.service is not None

    def submit_location(self, origin_lat: float, origin_lng: float,
                        dest_lat: float, dest_lng: float) -> str:
        """Submit location to service"""
        if not self.service:
            raise Exception("Service not started")

        self.request_count += 1
        request_id = f"req_{self.request_count}"

        user_location = UserLocation(
            user_id=1,  # Single user
            origin_lat=origin_lat,
            origin_lng=origin_lng,
            destination_lat=dest_lat,
            destination_lng=dest_lng,
            stored_at=datetime.now(timezone.utc),
        )

        self.service.add_user_location(user_location)
        return request_id

    def set_result_callback(self, callback: Callable):
        """Set callback for handling results"""
        self.result_callback = callback

    def _handle_service_result(self, result):
        """Handle result from clustering service"""
        print(f"📊 Service result: {result}")
        if self.result_callback:
            self.result_callback(result)

import threading
import time
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
from .data_ingestion import DataIngestionFactory, DataSourceType
from .data_storage import DataStorage
from .engine.MPBucketingEngine import ClusteringEngine
from .output_handlers import OutputHandlerFactory, OutputHandler
from .models import OutputMessage, UserLocation
from ..config import PLACE

logger = logging.getLogger(__name__)


class ClusteringService:
    """Main clustering service that runs in background threads"""

    def __init__(self, clustering_interval: int = 5, max_workers: int = 3):
        self.clustering_interval = clustering_interval
        self.max_workers = max_workers


        # Initialize components
        self.data_storage = DataStorage()
        self.clustering_engine = ClusteringEngine(place=PLACE)
        self.output_handlers: List[OutputHandler] = []

        # Threading
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="clustering"
        )

        # Data ingestion
        self.data_ingestors = []

        # UserID generator
        self.next_user_id = 1


    def add_data_ingestor(self, ingestor_type: DataSourceType, config: Dict) -> None:
        """Add a data ingestor to the service"""
        ingestor = DataIngestionFactory.create_ingestor(ingestor_type, config)
        self.data_ingestors.append(ingestor)

    def add_output_handler(self, handler_config: Dict) -> None:
        """Add an output handler to the service"""
        handler = OutputHandlerFactory.create_handlers([handler_config])[0]
        self.output_handlers.append(handler)

    def add_output_handlers(self, handler_configs: List[Dict]) -> None:
        """Add multiple output handlers to the service"""
        handlers = OutputHandlerFactory.create_handlers(handler_configs)
        self.output_handlers.extend(handlers)

    def _ingest_data(self) -> None:
        """Ingest data from all configured sources"""
        for ingestor in self.data_ingestors:
            try:
                data = ingestor.get_data()
                for user_location in data:
                    self.data_storage.add_user_location(user_location)
            except Exception as e:
                logger.error(
                    f"Error ingesting data from {ingestor.source_type}: {e}")

    def _process_output(self, message: OutputMessage) -> None:
        """Process outpuDATABASEt through all registered handlers"""
        for handler in self.output_handlers:
            try:
                handler.handle_output(message)
            except Exception as e:
                logger.error(f"Error in output handler: {e}")

    def _clustering_worker(self) -> None:
        """Main clustering worker that runs in background thread"""
        logger.info("Clustering worker started")

        while not self._stop_event.is_set():
            try:
                # Ingest new data
                self._ingest_data()

                # Get pending users
                pending_users = self.data_storage.get_pending_users()
                if len(pending_users) >= 3:
                    # Perform clustering
                    start_time = time.time()
                    new_groups = self.clustering_engine.cluster_users(
                        pending_users)
                    end_time = time.time()

                    logger.info(f"Clustering completed in {end_time - start_time:.4f} seconds, "
                                f"formed {len(new_groups)} groups")

                    # Process new groups
                    for group in new_groups:
                        self.data_storage.add_cluster_group(group)

                        # Create output message
                        message = OutputMessage(
                            message_type="group_formed",
                            data={
                                'group': group,
                                "group_id": group.group_id,
                                "users": group.get_user_ids(),
                                "group_data": group.to_dict()
                            }
                        )

                        # Process output
                        self._process_output(message)

                # # Process any updates (user removals, etc.)
                # while self.data_storage.has_updates():
                #     update = self.data_storage.get_update(block=False)
                #     if update:
                #         message = OutputMessage(
                #             message_type=update["type"],
                #             data=update
                #         )
                #         self._process_output(message)

            except Exception as e:
                logger.error(f"Error in clustering worker: {e}")

            # Wait for next iteration or stop signal
            self._stop_event.wait(self.clustering_interval)

        logger.info("Clustering worker stopped")

    def start(self) -> None:
        """Start the clustering service"""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Clustering service is already running")
            return

        # Setup data ingestors
        for ingestor in self.data_ingestors:
            ingestor.setup()

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._clustering_worker,
            daemon=True,
            name="ClusteringServiceWorker"
        )
        self._thread.start()
        logger.info("Clustering service started")

    def stop(self) -> None:
        """Stop the clustering service"""
        if self._thread is None:
            return

        logger.info("Stopping clustering service...")
        self._stop_event.set()

        if self._thread.is_alive():
            self._thread.join(timeout=10)

        # Teardown data ingestors
        for ingestor in self.data_ingestors:
            ingestor.teardown()

        self._executor.shutdown(wait=False)
        logger.info("Clustering service stopped")

    def add_user_location(self, user_location: UserLocation) -> None:
        """Add a user location directly to the service"""
        self.data_storage.add_user_location(user_location)

    def remove_user(self, user_id: int) -> bool:
        """Remove a user from the service"""
        return self.data_storage.remove_user_location(user_id)

    # Public API methods
    def get_user_group(self, user_id: int) -> Optional[Dict]:
        """Get group information for a specific user"""
        group = self.data_storage.get_group_by_user(user_id)
        return group.to_dict() if group else None

    def get_user_companions(self, user_id: int) -> Optional[Dict]:
        """Get companions for a specific user"""
        group = self.data_storage.get_group_by_user(user_id)
        if not group:
            return None

        companions = [u for u in group.users if u.user_id != user_id]
        return {
            "group_id": group.group_id,
            "companions": [comp.to_dict() for comp in companions],
            "meeting_point_origin": group.meeting_point_origin,
            "meeting_point_destination": group.meeting_point_destination,
            "created_at": group.created_at.isoformat(),
        }

    def get_all_active_groups(self) -> List[Dict]:
        """Get all active groups"""
        with self.data_storage._lock:
            return [group.to_dict() for group in self.data_storage.cluster_groups.values()]

    def get_service_status(self) -> Dict:
        """Get current service status"""
        with self.data_storage._lock:
            return {
                "is_running": self._thread is not None and self._thread.is_alive(),
                "active_groups": len(self.data_storage.cluster_groups),
                "complete_groups": len(
                    [g for g in self.data_storage.cluster_groups.values()
                     if g.is_complete()]
                ),
                "total_users": len(self.data_storage.user_locations),
                "users_in_groups": len(self.data_storage.user_to_group),
                "clustering_interval": self.clustering_interval,
            }
        
    def get_all_users(self)-> List[UserLocation]: 
        """Get all active users"""
        return [user.to_dict() for user in self.data_storage.get_all_users()]
    
    def get_next_user_id(self)-> int:
        """Get user id for new user"""
        user_id = self.next_user_id
        self.next_user_id += 1
        return user_id


_clustering_service: ClusteringService | None = None


def get_clustering_service(clustering_interval=5) -> ClusteringService:
    global _clustering_service

    if _clustering_service is None:
        _clustering_service = ClusteringService(
            clustering_interval=clustering_interval)

    return _clustering_service

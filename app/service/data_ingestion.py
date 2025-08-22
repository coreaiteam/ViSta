from abc import ABC, abstractmethod
from typing import List, Dict
from .models import UserLocation, DataSourceType
import logging

logger = logging.getLogger(__name__)


class DataIngestor(ABC):
    """Abstract base class for data ingestion"""

    def __init__(self, source_type: DataSourceType):
        self.source_type = source_type

    @abstractmethod
    def get_data(self) -> List[UserLocation]:
        """Retrieve data from the source"""
        pass

    @abstractmethod
    def setup(self):
        """Setup the data source connection"""
        pass

    @abstractmethod
    def teardown(self):
        """Teardown the data source connection"""
        pass


class DatabaseIngestor(DataIngestor):
    """Ingest data from a database"""

    def __init__(self, db_session_factory):
        super().__init__(DataSourceType.DATABASE)
        self.db_session_factory = db_session_factory

    def setup(self):
        logger.info("Setting up database ingestor")
        # Connection is established on demand

    def teardown(self):
        logger.info("Tearing down database ingestor")
        # Connection is closed after each use

    def get_data(self) -> List[UserLocation]:
        # Import here to avoid circular imports
        from app.database.models import Location

        session = self.db_session_factory()
        try:
            locations = session.query(Location).all()
            return [
                UserLocation(
                    user_id=loc.user_id,
                    origin_lat=loc.origin_lat,
                    origin_lng=loc.origin_lng,
                    destination_lat=loc.destination_lat,
                    destination_lng=loc.destination_lng,
                    stored_at=loc.stored_at,
                )
                for loc in locations
            ]
        except Exception as e:
            logger.error(f"Error fetching data from database: {e}")
            return []
        finally:
            session.close()


class StreamIngestor(DataIngestor):
    """Ingest data from a stream (e.g., Kafka, RabbitMQ)"""

    def __init__(self, stream_config: Dict):
        super().__init__(DataSourceType.STREAM)
        self.stream_config = stream_config
        self.connected = False

    def setup(self):
        logger.info("Setting up stream ingestor")
        # Implement stream connection logic
        self.connected = True

    def teardown(self):
        logger.info("Tearing down stream ingestor")
        # Implement stream disconnection logic
        self.connected = False

    def get_data(self) -> List[UserLocation]:
        if not self.connected:
            self.setup()

        # Implement stream consumption logic
        # This would typically be event-driven rather than polling
        # For now, return empty list as this would be implemented differently
        return []


class DataIngestionFactory:
    """Factory to create appropriate data ingestor"""

    @staticmethod
    def create_ingestor(ingestor_type: DataSourceType, config: Dict) -> DataIngestor:
        if ingestor_type == DataSourceType.DATABASE:
            return DatabaseIngestor(config.get('session_factory'))
        elif ingestor_type == DataSourceType.STREAM:
            return StreamIngestor(config.get('stream_config', {}))
        else:
            raise ValueError(f"Unsupported ingestor type: {ingestor_type}")

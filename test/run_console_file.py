# run_console_file.py
import logging
from app.service.service import ClusteringService
from app.service.data_ingestion import DataSourceType
from .file_output_handler import FileOutputHandler
from app.database.base import SessionLocal

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def console_callback(message):
    """Simple console output handler"""
    logger.info(f"Received {message.message_type}: {message.data}")


def main():
    # Create service
    service = ClusteringService(clustering_interval=5)

    # Add database ingestor
    service.add_data_ingestor(
        DataSourceType.DATABASE,
        {"session_factory": SessionLocal}
    )

    # Add console callback handler
    service.add_output_handler({
        "type": "callback",
        "callback": console_callback
    })

    # Add file output handler
    file_handler = FileOutputHandler("test/logs/clustering_events.log")
    service.add_output_handler({
        "type": "callback",
        "callback": file_handler.handle_output
    })

    # Start the service
    service.start()
    logger.info("Service started with console and file output")

    try:
        # Keep the service running
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down service...")
        service.stop()


if __name__ == "__main__":
    main()

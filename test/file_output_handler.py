# file_output_handler.py
import json
import logging
from datetime import datetime
from pathlib import Path
from app.service.output_handlers import OutputHandler
from app.service.models import OutputMessage

logger = logging.getLogger(__name__)


class FileOutputHandler(OutputHandler):
    """Output handler that writes messages to a file"""

    def __init__(self, file_path: str, max_file_size: int = 10485760):  # 10MB default
        self.file_path = Path(file_path)
        self.max_file_size = max_file_size
        self._ensure_file_exists()

    def _ensure_file_exists(self):
        """Create the file if it doesn't exist"""
        if not self.file_path.exists():
            self.file_path.parent.mkdir(parents=True, exist_ok=True)
            self.file_path.touch()

    def _rotate_file_if_needed(self):
        """Rotate file if it exceeds the maximum size"""
        if self.file_path.stat().st_size > self.max_file_size:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = self.file_path.parent / \
                f"{self.file_path.stem}_{timestamp}{self.file_path.suffix}"
            self.file_path.rename(backup_path)
            self.file_path.touch()

    def handle_output(self, message: OutputMessage) -> None:
        try:
            self._rotate_file_if_needed()

            # Format the message with timestamp
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": message.message_type,
                "data": message.data
            }

            # Append to file
            with open(self.file_path, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

            logger.info(f"Logged message to file: {message.message_type}")

        except Exception as e:
            logger.error(f"Error writing to file: {e}")

from abc import ABC, abstractmethod
from typing import Callable, List, Any, Dict
from .models import OutputMessage
import logging
import asyncio

logger = logging.getLogger(__name__)


class OutputHandler(ABC):
    """Abstract base class for output handlers"""

    @abstractmethod
    def handle_output(self, message: OutputMessage) -> None:
        """Handle an output message"""
        pass


class CallbackOutputHandler(OutputHandler):
    """Output handler that uses callbacks"""

    def __init__(self, callback: Callable[[OutputMessage], Any]):
        self.callback = callback

    def handle_output(self, message: OutputMessage) -> None:
        try:
            self.callback(message)
        except Exception as e:
            logger.error(f"Error in callback handler: {e}")


class WebSocketOutputHandler(OutputHandler):
    """Output handler that sends messages via WebSocket"""

    def __init__(self, connection_manager: Any):
        self.connection_manager = connection_manager

    def handle_output(self, message: OutputMessage) -> None:
        try:
            # Send the message
            if message.message_type == "group_formed":
                self._send_group_message(message)

        except Exception as e:
            logger.error(f"Error in WebSocket handler: {e}")

    def _send_group_message(self, message: OutputMessage) -> None:
        """Send group message to specific users"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        group_data = message.data
        group = group_data.get("group", None)

        for user in group.users:
            # Get companions for this user (excluding themselves)
            companions = [u for u in group.users if u.user_id != user.user_id]

            message = {
                'type': 'group_formed',
                'group_id': group.group_id,
                'companions': [
                    {
                        'user_id': comp.user_id,
                        'origin_lat': comp.origin_lat,
                        'origin_lng': comp.origin_lng,
                        'destination_lat': comp.destination_lat,
                        'destination_lng': comp.destination_lng,
                        'stored_at': comp.stored_at.isoformat()
                    }
                    for comp in companions
                ],
                'meeting_point_origin': group.meeting_point_origin,
                'meeting_point_destination': group.meeting_point_destination,
                'created_at': group.created_at.isoformat()
            }

            # Send to specific user
            loop.run_until_complete(
                self.connection_manager.send_personal_message(
                    user.user_id, message
                )
            )


class APIOutputHandler(OutputHandler):
    """Output handler that stores data for API endpoints"""

    def __init__(self, api_state_manager: Any):
        self.api_state_manager = api_state_manager

    def handle_output(self, message: OutputMessage) -> None:
        self.api_state_manager.update_state(message)


class OutputHandlerFactory:
    """Factory to create output handlers"""

    @staticmethod
    def create_handlers(handler_configs: List[Dict]) -> List[OutputHandler]:
        handlers = []
        for config in handler_configs:
            handler_type = config.get("type")
            if handler_type == "callback":
                handlers.append(CallbackOutputHandler(config["callback"]))
            elif handler_type == "websocket":
                handlers.append(WebSocketOutputHandler(
                    config["connection_manager"]))
            elif handler_type == "api":
                handlers.append(APIOutputHandler(config["api_state_manager"]))
        return handlers

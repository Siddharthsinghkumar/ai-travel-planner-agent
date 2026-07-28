import signal
import threading
import logging
from typing import Callable, List

logger = logging.getLogger(__name__)

class SignalHandler:
    def __init__(self):
        self.shutdown_requested = threading.Event()
        self._shutdown_callbacks: List[Callable] = []

    def setup(self):
        """Registers the signal handlers with the OS. Should be called on main thread."""
        try:
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            logger.info("Signal handlers for SIGINT and SIGTERM registered.")
        except Exception as e:
            logger.warning(f"Could not register signal handlers: {e}")

    def _handle_signal(self, signum, frame):
        logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
        self.shutdown_requested.set()
        
        # Execute callbacks (e.g., to trigger LangGraph checkpointing)
        for callback in self._shutdown_callbacks:
            try:
                callback()
            except Exception as e:
                logger.error(f"Error during shutdown callback: {e}", exc_info=True)

    def is_shutdown_requested(self) -> bool:
        return self.shutdown_requested.is_set()

    def register_shutdown_callback(self, callback: Callable):
        """Register a function to be called when shutdown is requested.
        Useful for triggering state checkpoints."""
        if callback not in self._shutdown_callbacks:
            self._shutdown_callbacks.append(callback)

# Global instance
global_signal_handler = SignalHandler()

def setup_signal_handlers():
    global_signal_handler.setup()

def is_shutdown_requested() -> bool:
    return global_signal_handler.is_shutdown_requested()

def register_shutdown_callback(callback: Callable):
    global_signal_handler.register_shutdown_callback(callback)

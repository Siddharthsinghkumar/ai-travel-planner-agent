import signal
from core.signal_handler import SignalHandler

def test_signal_handler_sets_flag():
    handler = SignalHandler()
    assert not handler.is_shutdown_requested()
    
    # Simulate a signal call
    handler._handle_signal(signal.SIGINT, None)
    
    assert handler.is_shutdown_requested()

def test_signal_handler_callbacks():
    handler = SignalHandler()
    
    callback_executed = False
    def mock_checkpoint_callback():
        nonlocal callback_executed
        callback_executed = True
        
    handler.register_shutdown_callback(mock_checkpoint_callback)
    
    # Simulate signal
    handler._handle_signal(signal.SIGTERM, None)
    
    assert handler.is_shutdown_requested()
    assert callback_executed is True

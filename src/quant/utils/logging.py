import logging
import os
import datetime
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    log_subdir: str = "general",
    log_level: int = logging.INFO,
    base_log_dir: str = "logs",
    include_console: bool = True,
    console_level: Optional[int] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_subdir: Subdirectory within logs/ for this logger
        log_level: Logging level for file handler
        base_log_dir: Base directory for log files
        include_console: Whether to add console handler
        console_level: Console logging level (defaults to log_level)
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(log_level)
    
    # Create log directory
    log_dir = Path(base_log_dir) / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create timestamped log file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{ts}.log"
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler (optional)
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level or log_level)
        console_formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger


def get_script_logger(script_name: Optional[str] = None) -> logging.Logger:
    """
    Convenience function to get a logger for a script.
    Automatically determines subdirectory from script name.
    """
    if script_name is None:
        # Try to get the calling script's name
        import inspect
        frame = inspect.currentframe().f_back
        script_name = Path(frame.f_globals.get('__file__', 'unknown')).stem
    
    log_subdir = script_name.replace('_', '/').replace('-', '/')
    return setup_logger(
        name=script_name,
        log_subdir=log_subdir
    )


# Context manager version (if you prefer this approach)
class LoggerContext:
    """Context manager for temporary logging setup."""
    
    def __init__(self, name: str, log_subdir: str = "general", **kwargs):
        self.name = name
        self.log_subdir = log_subdir
        self.kwargs = kwargs
        self.logger = None
    
    def __enter__(self):
        self.logger = setup_logger(self.name, self.log_subdir, **self.kwargs)
        self.logger.info(f"{self.name} script initialized")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.logger.error(f"Script ended with error: {exc_val}")
        else:
            self.logger.info(f"{self.name} script completed successfully")
        
        # Clean up handlers to avoid memory leaks
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
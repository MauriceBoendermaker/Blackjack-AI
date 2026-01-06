"""
Centralized logging system with print() redirection and categorization
"""

import sys
import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Tuple, Optional


class LogCategory(Enum):
    """Log categories for organizing console output"""
    MODEL_LOADING = "Model Loading"
    CARD_COUNTER = "Card Counter"
    PERFORMANCE = "Performance"
    ERROR = "Error"
    STARTUP = "Startup"
    CACHE = "Cache"
    UI_WARNING = "UI Warning"
    GENERAL = "General"


@dataclass
class LogEntry:
    """Single log entry with metadata"""
    timestamp: datetime
    category: LogCategory
    message: str
    level: str  # "INFO", "WARNING", "ERROR"


class LogManager:
    """
    Singleton log manager that captures and categorizes all console output
    Thread-safe with callback support for real-time updates
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        # Thread-safe log storage with circular buffer
        self.logs = deque(maxlen=1000)
        self.callbacks = []
        self.lock = threading.Lock()

        # Compile regex patterns for category detection
        self.patterns = self._compile_patterns()

    def _compile_patterns(self):
        """Compile regex patterns for efficient category detection"""
        return [
            # (pattern, category, level)
            (re.compile(r'⚠️.*Slow cycle', re.IGNORECASE), LogCategory.PERFORMANCE, 'WARNING'),
            (re.compile(r'Updated counter for|Updated modern widget'), LogCategory.CARD_COUNTER, 'INFO'),
            (re.compile(r'\[update_count\]'), LogCategory.CARD_COUNTER, 'INFO'),
            (re.compile(r'Loaded.*recommendations|Saved.*recommendations'), LogCategory.CACHE, 'INFO'),
            (re.compile(r'Reset for new round'), LogCategory.CACHE, 'INFO'),
            (re.compile(r'Failed to|Error|error'), LogCategory.ERROR, 'ERROR'),
            (re.compile(r'Initialized.*model|✓.*model'), LogCategory.MODEL_LOADING, 'INFO'),
            (re.compile(r'Using (local YOLO|Roboflow API|Roboflow workspace)'), LogCategory.MODEL_LOADING, 'INFO'),
            (re.compile(r'loading Roboflow'), LogCategory.MODEL_LOADING, 'INFO'),
            (re.compile(r'Starting|🎰 Loading|✓.*loaded successfully'), LogCategory.STARTUP, 'INFO'),
            (re.compile(r'\[UI Thread\]|\[Warning\] GUI'), LogCategory.UI_WARNING, 'WARNING'),
            (re.compile(r'\[Card Counters\]'), LogCategory.CARD_COUNTER, 'INFO'),
            (re.compile(r'P\d+:.*cards|Card value'), LogCategory.CARD_COUNTER, 'INFO'),
        ]

    def _categorize(self, message: str) -> Tuple[LogCategory, str]:
        """
        Detect category and level from message content
        Returns: (category, level)
        """
        for pattern, category, level in self.patterns:
            if pattern.search(message):
                return category, level

        # Default to GENERAL INFO
        return LogCategory.GENERAL, 'INFO'

    def add_log(self, message: str):
        """
        Add a new log entry
        Thread-safe with callback notifications
        """
        if not message.strip():
            return

        category, level = self._categorize(message)
        entry = LogEntry(
            timestamp=datetime.now(),
            category=category,
            message=message.strip(),
            level=level
        )

        with self.lock:
            self.logs.append(entry)

        # Notify all registered callbacks
        for callback in self.callbacks:
            try:
                callback(entry)
            except Exception as e:
                # Prevent callback errors from breaking logging
                print(f"Error in log callback: {e}", file=sys.__stdout__)

    def register_callback(self, callback: Callable[[LogEntry], None]):
        """Register a callback to be notified of new log entries"""
        if callback not in self.callbacks:
            self.callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[LogEntry], None]):
        """Unregister a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def get_all_logs(self):
        """Get a copy of all current logs (thread-safe)"""
        with self.lock:
            return list(self.logs)

    def clear_logs(self):
        """Clear all stored logs (thread-safe)"""
        with self.lock:
            self.logs.clear()


class PrintRedirector:
    """
    Redirects stdout to capture print() statements while preserving console output
    """
    def __init__(self, log_manager: LogManager, original_stdout):
        self.log_manager = log_manager
        self.original_stdout = original_stdout
        self._buffer = ""

    def write(self, text: str):
        """
        Intercept stdout writes
        Forward to original stdout and log manager
        """
        # Always write to original console
        self.original_stdout.write(text)
        self.original_stdout.flush()

        # Buffer partial lines
        self._buffer += text

        # Process complete lines
        if '\n' in self._buffer:
            lines = self._buffer.split('\n')
            # Last item might be incomplete, keep it in buffer
            self._buffer = lines[-1]

            # Send complete lines to log manager
            for line in lines[:-1]:
                if line.strip():
                    self.log_manager.add_log(line.strip())

    def flush(self):
        """Flush the stream"""
        self.original_stdout.flush()

    def isatty(self):
        """Check if the stream is a TTY"""
        return self.original_stdout.isatty()


# Global singleton access
_log_manager_instance = None


def get_log_manager() -> LogManager:
    """Get the global LogManager singleton"""
    global _log_manager_instance
    if _log_manager_instance is None:
        _log_manager_instance = LogManager()
    return _log_manager_instance

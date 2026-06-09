"""Centralized logging with print() redirection and categorization.

Thread-safety notes (this code runs on the Tk thread AND the worker thread):
  * the log deque, the callback list, and the redirector line buffer are all
    guarded by locks
  * callbacks are invoked on the CALLING thread — GUI consumers must marshal
    to the Tk thread themselves (LoggingWindow queues entries and drains them
    from an `after()` loop)
"""

import re
import sys
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Callable, Tuple


class LogCategory(Enum):
    MODEL_LOADING = "Model Loading"
    CARD_COUNTER = "Card Counter"
    DETECTION = "Detection"
    PERFORMANCE = "Performance"
    ERROR = "Error"
    STARTUP = "Startup"
    UI = "UI"
    GENERAL = "General"


@dataclass
class LogEntry:
    timestamp: datetime
    category: LogCategory
    message: str
    level: str  # "INFO" | "WARNING" | "ERROR"


_PATTERNS = [
    (re.compile(r"error|failed|exception|traceback", re.IGNORECASE), LogCategory.ERROR, "ERROR"),
    (re.compile(r"slow cycle|cycle_ms|performance", re.IGNORECASE), LogCategory.PERFORMANCE, "WARNING"),
    (re.compile(r"count|shoe|counter", re.IGNORECASE), LogCategory.CARD_COUNTER, "INFO"),
    (re.compile(r"model|roboflow|yolo|weights", re.IGNORECASE), LogCategory.MODEL_LOADING, "INFO"),
    (re.compile(r"^P\d+ card|dealer|round|cutting card|table cleared", re.IGNORECASE), LogCategory.DETECTION, "INFO"),
    (re.compile(r"starting|loaded|ready|initialized", re.IGNORECASE), LogCategory.STARTUP, "INFO"),
    (re.compile(r"monitor|window|status", re.IGNORECASE), LogCategory.UI, "INFO"),
]


class LogManager:
    """Singleton store of recent log entries with change callbacks."""

    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self.logs = deque(maxlen=2000)
        self._callbacks = []
        self._lock = threading.Lock()

    @staticmethod
    def _categorize(message: str) -> Tuple[LogCategory, str]:
        for pattern, category, level in _PATTERNS:
            if pattern.search(message):
                return category, level
        return LogCategory.GENERAL, "INFO"

    def add_log(self, message: str, level: str | None = None):
        message = message.strip()
        if not message:
            return
        category, detected_level = self._categorize(message)
        entry = LogEntry(datetime.now(), category, message, level or detected_level)
        with self._lock:
            self.logs.append(entry)
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback(entry)
            except Exception as e:
                print(f"Error in log callback: {e}", file=sys.__stdout__)

    def register_callback(self, callback: Callable[[LogEntry], None]):
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def unregister_callback(self, callback: Callable[[LogEntry], None]):
        with self._lock:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

    def get_all_logs(self):
        with self._lock:
            return list(self.logs)

    def clear_logs(self):
        with self._lock:
            self.logs.clear()


class PrintRedirector:
    """Wraps a std stream: console output is preserved, complete lines are
    forwarded to the LogManager. Safe when the original stream is None
    (pythonw.exe) and when multiple threads print concurrently."""

    def __init__(self, log_manager: LogManager, original, level: str | None = None):
        self.log_manager = log_manager
        self.original = original
        self.level = level
        self._buffer = ""
        self._lock = threading.Lock()

    def write(self, text: str):
        if self.original is not None:
            try:
                self.original.write(text)
            except Exception:
                pass
        lines_to_log = []
        with self._lock:
            self._buffer += text
            if "\n" in self._buffer:
                parts = self._buffer.split("\n")
                self._buffer = parts[-1]
                lines_to_log = [ln for ln in parts[:-1] if ln.strip()]
        for line in lines_to_log:
            self.log_manager.add_log(line, level=self.level)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass

    def isatty(self):
        return bool(self.original is not None and self.original.isatty())


def install_redirectors(log_manager: LogManager):
    """Route stdout and stderr through the log manager (console preserved)."""
    sys.stdout = PrintRedirector(log_manager, sys.stdout)
    sys.stderr = PrintRedirector(log_manager, sys.stderr, level="ERROR")

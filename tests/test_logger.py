"""Tests for src/murmur/logger.py"""

import logging
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestSetupLogger:
    """Test setup_logger function."""

    def test_setup_logger_returns_logger(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            result = logger_module.setup_logger()
            assert isinstance(result, logging.Logger)

    def test_setup_logger_name_is_murmur(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            result = logger_module.setup_logger()
            assert result.name == "murmur"

    def test_setup_logger_level_is_debug(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            result = logger_module.setup_logger()
            assert result.level == logging.DEBUG

    def test_setup_logger_creates_log_directory(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.setup_logger()
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            assert log_dir.exists()

    def test_setup_logger_has_file_handler(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger = logger_module.setup_logger()
            file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) >= 1

    def test_setup_logger_has_console_handler(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger = logger_module.setup_logger()
            stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
            assert len(stream_handlers) >= 1

    def test_setup_logger_console_handler_error_level(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger = logger_module.setup_logger()
            stream_handlers = [h for h in logger.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)]
            assert any(h.level == logging.ERROR for h in stream_handlers)

    def test_setup_logger_idempotent(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger1 = logger_module.setup_logger()
            handler_count = len(logger1.handlers)

            logger2 = logger_module.setup_logger()
            assert logger1 is logger2
            assert len(logger2.handlers) == handler_count

    def test_log_file_named_with_date(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.setup_logger()
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            today = datetime.now().strftime('%Y-%m-%d')
            expected_file = log_dir / f"murmur-{today}.log"
            assert expected_file.exists()


class TestLogModuleLevel:
    """Test module-level log object."""

    def test_log_is_logger_instance(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            assert isinstance(logger_module.log, logging.Logger)

    def test_log_can_write_debug(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.log.debug("Test debug message")
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"murmur-{today}.log"
            content = log_file.read_text()
            assert "Test debug message" in content

    def test_log_can_write_info(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.log.info("Test info message")
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"murmur-{today}.log"
            content = log_file.read_text()
            assert "Test info message" in content

    def test_log_can_write_error(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.log.error("Test error message")
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"murmur-{today}.log"
            content = log_file.read_text()
            assert "Test error message" in content

    def test_log_format_includes_timestamp(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.log.info("Timestamp test")
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"murmur-{today}.log"
            content = log_file.read_text()
            assert "[INFO]" in content

    def test_log_format_includes_level(self, temp_dir):
        with patch.object(Path, 'home', return_value=temp_dir):
            from murmur import logger as logger_module
            import importlib
            importlib.reload(logger_module)

            logger_module.log.warning("Warning test")
            log_dir = temp_dir / "Library" / "Logs" / "Murmur"
            today = datetime.now().strftime('%Y-%m-%d')
            log_file = log_dir / f"murmur-{today}.log"
            content = log_file.read_text()
            assert "[WARNING]" in content

# Standard Library
import sys
from unittest.mock import MagicMock

# Mock heavy libraries on Windows for CI tests
if sys.platform.startswith(("win")):
    # openslide
    mock_openslide = MagicMock()
    mock_openslide.OpenSlide = MagicMock()
    mock_openslide.deepzoom = MagicMock()
    sys.modules["openslide"] = mock_openslide
    sys.modules["openslide.deepzoom"] = mock_openslide.deepzoom

    # pyvips
    mock_pyvips = MagicMock()
    mock_pyvips.Image = MagicMock()
    sys.modules["pyvips"] = mock_pyvips

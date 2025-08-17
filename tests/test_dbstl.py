import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path
from database import DBController

def test_get_closest_from_db():
    # --- Prepare dummy data ---
    dummy_vec = np.random.rand(1, 128).astype(np.float32)
    dummy_ids = np.array([[1, 2, 3]])
    dummy_img_paths = {1: "img1.jpg", 2: "img3.jpg", 3: "img5.jpg"}

    # Mock get_vec to return dummy vector
    def mock_get_vec(path):
        return dummy_vec, None, None

    # Mock FAISS index
    mock_index = MagicMock()
    mock_index.search.return_value = (np.array([[0.1, 0.2, 0.3]]), dummy_ids)

    # Mock SQLAlchemy session
    mock_session = MagicMock()
    def mock_get(model, id_):
        class DummyImg:
            def __init__(self, path): self.path = path
        return DummyImg(dummy_img_paths.get(id_))

    mock_session.get.side_effect = mock_get

    mock_session_cm = MagicMock()
    mock_session_cm.__enter__.return_value = mock_session
    mock_session_cm.__exit__.return_value = False

    # Patch Session to return mock session context
    with patch("database.dbctl.Session", return_value=mock_session_cm):
        
        dbctl = DBController(db_path='pytest.db', index_path=Path('pytest_indes.faiss'), kmeans_path=Path('pytest_kmeans.faiss'), img_drive_path=Path('.'))
        dbctl.idx = mock_index
        dbctl.engine = MagicMock()
        dbctl.get_vec = staticmethod(mock_get_vec)
        img_path = Path("query.jpg")

        result = dbctl.get_closest_from_db(img_path, k=3)
        print('Result: ', result)

        assert result == ["img1.jpg", "img3.jpg", "img5.jpg"]
        
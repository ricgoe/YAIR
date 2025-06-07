from database import DBController
from pathlib import Path

db = DBController(Path("test.db"), Path('Index_db.faiss'), Path("images"), threads=4, estimated_load=540_000)
db.populate_db(n=100)
from database import DBController
from pathlib import Path

db = DBController(Path("test.db"),
                  Path('Index_db.faiss'),
                  Path("models/binflat.faiss"),
                  Path("/Users/richardgox/Documents/4. Semester/GuglLens/models/byol_256.pth"),
                  Path("/Volumes/Big Data/data/image_data"),
                  threads=4, estimated_load=200_000,
                  orb_length=1024, color_length=0, byol_length=0)
db.populate_db(2000)
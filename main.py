from database import DBController
from pathlib import Path

db = DBController(Path("ImageDB.db"),
                  Path('ImageIDX.faiss'),
                  Path("/raid/richard/GuglLens/sift_kmeans.faiss"),
                  Path("/raid/richard/image_data"),
                  threads=20, estimated_load=540_000,
                  feat_length=512, color_length=26, dino_length=384)
db.populate_db()
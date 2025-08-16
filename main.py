from database import DBController
from pathlib import Path

db = DBController(Path("test.db"),
                  Path('Index_db.faiss'),
                  Path("kmeans.faiss"),
                  Path("/raid/richard/checkpoints_256/byol_backbone_r50_e100.pth"),
                  Path("/raid/richard/image_data/pexels_image_dataset_v2"),
                  threads=30, estimated_load=200_000,
                  orb_length=500, color_length=26, byol_length=256)
db.populate_db()
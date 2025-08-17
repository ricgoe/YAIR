from sqlmodel import SQLModel, Field, create_engine, Session, select

class ImgEntry(SQLModel, table = True):
    """
    SQLModel table representing a single image entry in the database.

    Fields:
        id (int): Auto-incrementing primary key.
        path (str): Unique file path to the image.
        width (int): Width of the image in pixels.
        height (int): Height of the image in pixels.
    """
    id: int = Field(default=None, primary_key=True)
    path: str = Field(unique=True)
    width: int = Field(default=None)
    height: int = Field(default=None)
from sqlmodel import SQLModel, Field, create_engine, Session, select

class ImgEntry(SQLModel, table = True):
    id: int = Field(default=None, primary_key=True)
    path: str = Field(unique=True)
    width: int = Field(default=None)
    height: int = Field(default=None)
    #faiss_id: int

class ImgConvertFailure(Exception):
    def __init__(self, file, error):
        super().__init__(f"Failed to convert {file}\n\t at {error}")
        
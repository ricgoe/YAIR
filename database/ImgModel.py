from sqlmodel import SQLModel, Field, create_engine, Session, select

class ImgEntry(SQLModel, table = True):
    id: int = Field(default=None, primary_key=True)
    path: str
    width: int
    height: int
    #faiss_id: int
    

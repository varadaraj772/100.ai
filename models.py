from sqlmodel import SQLModel, Field

class Knowledge(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    text: str

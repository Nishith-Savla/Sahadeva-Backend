from pydantic import BaseModel


class Response(BaseModel):
    message: str | None
    detail: str | None

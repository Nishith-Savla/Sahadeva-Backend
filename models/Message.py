from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class MessageResponse(BaseModel):
    messages: list[Message]

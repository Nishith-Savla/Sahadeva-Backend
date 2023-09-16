import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from importlib.metadata import files
from pathlib import Path
from typing import Annotated

import mysql.connector
import openai
import requests
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeAudioLoader, PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pydantic import BaseModel

_ = load_dotenv(find_dotenv())  # read local .env file


def connect_to_database() -> mysql.connector.connection.MySQLConnection:
    return mysql.connector.connect(host=os.getenv('DB_HOST'), database=os.getenv('DB_NAME'),
                                   user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))


@asynccontextmanager
async def lifespan(_: FastAPI):
    global vector_db, db
    db = connect_to_database()
    vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY,
                       embedding_function=OpenAIEmbeddings())
    load_qa()
    yield
    db.close()


app = FastAPI(lifespan=lifespan)

openai.api_key = os.getenv('OPENAI_API_KEY')
origins = ["*"]
CHROMA_PERSIST_DIRECTORY = 'docs/chroma/'

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=150
)
qa: ConversationalRetrievalChain | None = None
vector_db: Chroma | None = None


# template = """Use the following pieces of context to answer the question at the end.
# If you don't know the answer, just say that you don't know, don't try to make up an answer.
# Use three sentences maximum. Keep the answer as concise as possible.
# Always say "Thanks for asking!" at the end of the answer.
#
# Context: {context}
# Question: {question}
# Helpful Answer:"""
# QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

def load_qa():
    global vector_db, qa
    retriever = vector_db.as_retriever(search_type="mmr", search_kwargs={"top_k": 5})
    qa = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(temperature=0),
        retriever=retriever,
        # return_source_documents=True,
        # return_generated_question=True,
    )


class Message(BaseModel):
    role: str
    content: str


@app.post("/")
def root(messages: list[Message] = "") -> dict[str, list[Message]]:
    if not messages:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Messages are required",
        )

    chat_history = []
    for i in range(0, len(messages) - 1, 2):
        chat_history.append((messages[i].content, messages[i + 1].content))
    message = qa.run(question=messages[-1].content, chat_history=chat_history)
    messages.append(Message(role="assistant", content=message))
    return {"messages": messages}


@app.post("/signup")
async def signup(username: Annotated[str, Form()], password: Annotated[str, Form()]) -> dict[
    str, str]:
    mysql_query = "SELECT * FROM users WHERE username = %s"
    with db.cursor() as cursor:
        cursor.execute(mysql_query, (username,))
        result = cursor.fetchone()
        if result:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Email already exists",
            )

    mysql_query = "INSERT INTO users (username, password) VALUES (%s, %s)"
    with db.cursor() as cursor:
        cursor.execute(mysql_query, (username, password))
        db.commit()

    return {"message": "Signup successful"}


@app.post("/login")
async def login(username: Annotated[str, Form()], password: Annotated[str, Form()]) -> dict[
    str, str]:
    mysql_query = "SELECT * FROM users WHERE username = %s AND password = %s"
    with db.cursor() as cursor:
        cursor.execute(mysql_query, (username, password))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Incorrect username or password",
            )
        _username, _password = result

        if username != _username or password != _password:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Incorrect username or password",
            )

    return {"message": "Login successful"}


@app.post("/upload")
def upload_documents(files: list[UploadFile]) -> dict[str, str]:
    docs = []
    for file in files:
        output_file = Path("docs/pdfs") / file.filename
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_file.write_bytes(file.file.read())
        docs.extend(PyPDFLoader(output_file.as_posix()).load())

    splits = text_splitter.split_documents(docs)
    vector_db.add_documents(splits)
    vector_db.persist()

    return {"message": f"Documents uploaded successfully"}


def encode_youtube_url(url) -> str:
    return url.lstrip("https://www.youtube.com/watch?v=") + ".txt"


@app.post("/images")
def upload_images(files: list[UploadFile]) -> dict[str, str]:
    docs = []

    for file in files:
        response = requests.post("https://api.ocr.space/parse/image",
                                 files={'file': (
                                     file.filename, file.file)},
                                 headers={
                                     'apikey': os.getenv('OCR_SPACE_API_KEY')
                                 })
        docs.append(response.json()['ParsedResults'][0]['ParsedText'])

    splits = text_splitter.split_text(*docs)
    vector_db.add_texts(splits)
    vector_db.persist()

    return {"message": f"Images uploaded successfully"}


@app.post("/youtube")
def load_youtube_transcript(url: str) -> dict[str, str]:
    youtube_audio_save_dir = Path("docs/youtube")
    youtube_audio_save_dir.mkdir(exist_ok=True, parents=True)

    loader = GenericLoader(
        YoutubeAudioLoader([url], youtube_audio_save_dir.as_posix()),
        OpenAIWhisperParser()
    )
    docs = loader.load()
    text_splitter.split_documents(docs)
    vector_db.add_documents(docs)
    vector_db.persist()

    return {"message": f"Transcript for {url} saved successfully"}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)

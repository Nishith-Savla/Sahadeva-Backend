from contextlib import asynccontextmanager
from http import HTTPStatus
from importlib.metadata import files
from pathlib import Path

import mysql.connector
import openai
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeAudioLoader, PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

from helper import *
from models.Message import Message, MessageResponse
from models.Response import Response
from models.User import UserSignup, UserLogin

load_dotenv()  # read local .env file


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
CHROMA_PERSIST_DIRECTORY = 'docs/chroma/'

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), retriever=retriever)


@app.post("/")
def root(messages: list[Message]) -> MessageResponse:
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
    return MessageResponse(messages=messages)


@app.post("/signup")
async def signup(user_signup: UserSignup) -> Response:
    mysql_query = "SELECT * FROM users WHERE email = %s"
    with db.cursor() as cursor:
        cursor.execute(mysql_query, (user_signup.email,))
        result = cursor.fetchone()
        if result:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST,
                detail="Email already exists",
            )

    mysql_query = "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)"
    with db.cursor() as cursor:
        cursor.execute(mysql_query, (user_signup.name, user_signup.email, user_signup.password))
        db.commit()

    return Response(message="Signup successful")


@app.post("/login")
async def login(user_login: UserLogin) -> Response:
    mysql_query = "SELECT email, password FROM users WHERE email = %s AND password = %s"
    with db.cursor() as cursor:
        cursor.execute(mysql_query, (user_login.email, user_login.password))
        result = cursor.fetchone()
        if not result:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Incorrect email or password",
            )
        email, password = result

        if user_login.email != email or user_login.password != password:
            raise HTTPException(
                status_code=HTTPStatus.UNAUTHORIZED,
                detail="Incorrect email or password",
            )

    return Response(message="Login successful")


@app.post("/upload")
def upload_documents(files: list[UploadFile]) -> Response:
    docs = []
    for file in files:
        output_file = Path("docs/pdfs") / file.filename
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_file.write_bytes(file.file.read())
        docs.extend(PyPDFLoader(output_file.as_posix()).load())

    splits = text_splitter.split_documents(docs)
    vector_db.add_documents(splits)
    vector_db.persist()

    return Response(message=f"{len(files)} document(s) uploaded successfully")


@app.post("/images")
def upload_images(files: list[UploadFile]) -> Response:
    docs = []

    for file in files:
        parsed_text = extract_text_from_image(file.file, file.filename)
        docs.append(parsed_text)

    splits = text_splitter.split_text(*docs)
    vector_db.add_texts(splits)
    vector_db.persist()

    return Response(message=f"{len(files)} images(s) uploaded successfully")


@app.post("/youtube")
def load_youtube_transcript(url: str) -> Response:
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

    return Response(message=f"Transcript for {url} saved successfully")


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)

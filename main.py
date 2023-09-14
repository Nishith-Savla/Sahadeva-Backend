import os
from contextlib import asynccontextmanager
from http import HTTPStatus
from pathlib import Path
from typing import Any

import openai
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.base import Chain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import YoutubeAudioLoader, PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma

_ = load_dotenv(find_dotenv())  # read local .env file


@asynccontextmanager
async def lifespan(_: FastAPI):
    global vector_db
    vector_db = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY,
                       embedding_function=OpenAIEmbeddings())
    load_qa()
    yield


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


@app.post("/")
def root(messages: list[dict[str, str]] = "") -> dict[str, Any]:
    global qa
    if not messages:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Messages are required",
        )

    chat_history = []
    for i in range(0, len(messages) - 1, 2):
        chat_history.append((messages[i]['content'], messages[i + 1]['content']))
    message = qa.run(question=messages[-1]['content'], chat_history=chat_history)
    messages.append({"role": "assistant", "content": message})
    return {"messages": messages}


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


@app.get("/youtube")
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

    # output_file = Path("docs/youtube-transcripts") / encode_youtube_url(url)
    # output_file.parent.mkdir(exist_ok=True, parents=True)
    # for doc in docs:
    #     output_file.write_text(doc.page_content)
    #
    return {"message": f"Transcript for {url} saved successfully"}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)

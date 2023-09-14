import os
from http import HTTPStatus
from pathlib import Path
from typing import Any

import openai
import uvicorn
from dotenv import load_dotenv, find_dotenv
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.document_loaders import YoutubeAudioLoader, DirectoryLoader, UnstructuredPDFLoader, \
    TextLoader, PyPDFLoader
from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import OpenAIWhisperParser
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter

_ = load_dotenv(find_dotenv())  # read local .env file
app = FastAPI()

openai.api_key = os.getenv('OPENAI_API_KEY')
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

context = {
    "role": "system",
    "content": """
You are a Technical Query chatbot named `Sahadeva`, an automated service to help users find \
answers to their queries based on the context provided.

NOTE: Don't include any information out of the given context.

```

You respond in a short, very conversational friendly style.
""",
}


def get_completion_from_messages(messages: list[dict[str, str]], model: str = "gpt-3.5-turbo",
                                 temperature: float = 0.3) -> dict[str]:
    response = openai.ChatCompletion.create(
        model=model,
        messages=[context, *messages],
        temperature=temperature,
    )
    return response.choices[0].message


@app.post("/")
def root(messages: list[dict[str, str]] = "") -> dict[str, Any]:
    if not messages:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Messages are required",
        )

    index = VectorstoreIndexCreator(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    ).from_loaders(
        [
            DirectoryLoader("docs/PDFs", loader_cls=PyPDFLoader),
            # GenericLoader(FileSystemBlobLoader("docs/youtube", glob="*.m4a"),
            #               OpenAIWhisperParser()),
            DirectoryLoader("docs/youtube-transcripts", loader_cls=TextLoader),
        ])

    print("Index Created")
    return {"message": index.query(messages[-1]['content'])}


@app.post("/upload")
def upload_documents(files: list[UploadFile]) -> dict[str, str]:
    for file in files:
        output_file = Path("docs/PDFs") / file.filename
        output_file.parent.mkdir(exist_ok=True, parents=True)
        output_file.write_bytes(file.file.read())

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
    page_content = loader.load()[0].page_content
    print(page_content)

    output_file = Path("docs/youtube-transcripts") / encode_youtube_url(url)
    output_file.parent.mkdir(exist_ok=True, parents=True)
    output_file.write_text(page_content)

    return {"message": f"Transcript for {url} saved successfully"}


if __name__ == '__main__':
    uvicorn.run(app, host="localhost", port=8000)

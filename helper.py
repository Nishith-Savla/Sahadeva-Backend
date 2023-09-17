from typing import BinaryIO

import requests
import os


def extract_text_from_image(file: BinaryIO, filename: str):
    response = requests.post("https://api.ocr.space/parse/image",
                             files={'file': (
                                 filename, file)},
                             headers={
                                 'apikey': os.getenv('OCR_SPACE_API_KEY')
                             })
    return response.json()['ParsedResults'][0]['ParsedText']


def encode_youtube_url(url) -> str:
    return url.lstrip("https://www.youtube.com/watch?v=") + ".txt"



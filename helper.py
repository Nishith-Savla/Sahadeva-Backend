import logging
from time import sleep
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
    json = response.json()
    if isinstance(json, str):
        logging.info("OCR Space API limit reached, waiting 60 seconds")
        sleep(60)
        return extract_text_from_image(file, filename)
    try:
        return json['ParsedResults'][0]['ParsedText']
    except Exception as e:
        logging.info(json)
        logging.exception(e)
        return ""


def encode_youtube_url(url) -> str:
    return url.lstrip("https://www.youtube.com/watch?v=") + ".txt"



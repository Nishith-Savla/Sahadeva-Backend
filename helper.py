import logging
import os
from itertools import chain
from pathlib import Path
from time import sleep
from typing import BinaryIO

import bcrypt
import mysql.connector
import requests
from moviepy.video.io.VideoFileClip import VideoFileClip
from mysql.connector import MySQLConnection
from mysql.connector.pooling import PooledMySQLConnection
from scenedetect import AdaptiveDetector, detect, open_video, save_images


def connect_to_database() -> PooledMySQLConnection | MySQLConnection | None:
    return mysql.connector.connect(host=os.getenv('DB_HOST'), database=os.getenv('DB_NAME'),
                                   user=os.getenv('DB_USER'), password=os.getenv('DB_PASS'))


def extract_text_from_image(file: BinaryIO, filename: str) -> str:
    """
    Extracts text from image using OCR Space API
    Args:
        file: Image file object
        filename: Image filename

    Returns:
        str: parsed text from image
    """
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
        logging.exception(json, exc_info=e)
        return ""


def extract_text_from_video(output_file_path):
    """
    Extracts text from video using OpenAI's Whisper parser
    Args:
        output_file_path: path to video file

    Returns:
        list[str]: list of parsed text from video

    """
    docs = []
    scene_list = detect(output_file_path, AdaptiveDetector(adaptive_threshold=50))
    frames = chain.from_iterable(save_images(scene_list, open_video(output_file_path),
                                             output_dir="docs/frames/").values())
    for frame_path in frames:
        with open("docs/frames/" + frame_path, 'rb') as frame:
            parsed_text = extract_text_from_image(frame, frame_path)
        docs.append(parsed_text)
        Path("docs/frames/" + frame_path).unlink()
    return docs


def encode_youtube_url(url: str, ext: str = ".txt") -> str:
    """Converts YouTube url to a filename by extracting its id
    Args:
        url: YouTube URL
        ext: The output file extension

    Returns:
        str: filename with extension
    """
    return url.lstrip("https://www.youtube.com/watch?v=") + ext


def extract_audio_from_video(video_file: str, output_ext="mp3") -> str:
    """
    Converts video to audio using MoviePy library that uses `ffmpeg` under the hood
    Args:
        video_file: path to video file
        output_ext: output audio file extension

    Returns:
        str: path to audio file
    """
    filename, ext = os.path.splitext(video_file)
    with VideoFileClip(video_file) as clip:
        clip.audio.write_audiofile(f"{filename}.{output_ext}")
    return f"{filename}.{output_ext}"


def get_hashed_password(plain_text_password: str) -> bytes:
    # Hash a password for the first time
    #   (Using bcrypt, the salt is saved into the hash itself)
    return bcrypt.hashpw(plain_text_password.encode(), bcrypt.gensalt())


def is_correct_password(plain_text_password: str, hashed_password: bytes) -> bool:
    # Check hashed password. Using bcrypt, the salt is saved into the hash itself
    return bcrypt.checkpw(plain_text_password.encode(), hashed_password)

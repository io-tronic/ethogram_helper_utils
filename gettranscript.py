#!/usr/bin/env python

import os
import sys
import openai
from pydub import AudioSegment
import webvtt
from datetime import datetime, timedelta
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

CHUNK_TIME = 120

if not os.path.exists("temp_files"):
    os.mkdir("temp_files")


def make_slices(audio_path):
    track = AudioSegment.from_file(audio_path)

    # PyDub handles time in milliseconds
    chunk_len = CHUNK_TIME * 1000

    slices = track[::chunk_len]

    i = 0
    paths = []
    for slice in slices:
        slice.export("temp_files/slice_" + str(i) + ".wav", format="wav")
        paths.append("temp_files/slice_" + str(i))
        i += 1

    return paths


def get_slice_transcription(audio_path, style_prompt, last):

    prompt = style_prompt + " ".join(last.split()[-100:])
    print(prompt)
    print(audio_path)

    transcription = None
    with open(audio_path, "rb") as audio_file:
        transcription = openai.Audio.transcribe(
            "whisper-1", audio_file, response_format="vtt", prompt=prompt
        )

    return transcription


def adjust_caption_time(timestamp, seconds):
    t = datetime.strptime(timestamp, "%H:%M:%S.%f")
    d = timedelta(seconds=seconds)
    new_timestamp = t + d

    return new_timestamp.strftime("%H:%M:%S.%f")

def main(audio_path):

    paths = make_slices(audio_path)
    print(paths)
    style_prompt = (
        "um... so, yeah i understand, mhm. --- ok, let's move on, huh. --- ok. ---"
    )


    full_vtt = webvtt.WebVTT()
    full_text = ""
    caption_to_json = []

    i = 0
    for path in paths:

        with open(path + ".vtt", "w") as sub_file:
            transcription = get_slice_transcription(path + ".wav", style_prompt, full_text)
            sub_file.write(transcription)

        vtt = webvtt.read(path + ".vtt")
        for caption in vtt:

            # increment captions
            caption.start = adjust_caption_time(caption.start, CHUNK_TIME * i)
            caption.end = adjust_caption_time(caption.end, CHUNK_TIME * i)

            # add caption text to transcript file
            full_text = full_text + caption.text + "\n"

            caption_to_json.append([caption.text, caption.start, caption.end])

            full_vtt.captions.append(caption)

        print(transcription)

        i += 1

    new_filepath = audio_path.split('.')[0].split('_')[0]

    with open(new_filepath + "_transcript.vtt", "w") as fd:
        full_vtt.write(fd)

    with open(new_filepath + "_transcript.txt", "w") as fd:
        fd.write(full_text)

    with open(new_filepath + "_transcript.json", "w") as fd:
        fd.write(json.dumps(caption_to_json, indent=2))


if __name__ == "__main__":
    main(sys.argv[1])
#!/usr/bin/env python


import os
import sys
import openai
import json
import re
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = os.getenv("OPENAI_API_KEY")

#exponential backoff
@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)

def sliding_window(data, window_size, step_size, path):
    # Now, using the window_size and step_size, divide your data into overlapping chunks
    chunks = [data[i: i + window_size] for i in range(0, len(data) - window_size + 1, step_size)]

    chunk_context = [data[i: i + window_size*3] for i in range(0, len(data) - window_size*3 + 1, step_size)]

    last_pair = {"Q":None,"A":None}

    qa_pairs = []
    

    # Now feed these chunks into your ML function
    i=0
    for chunk in chunks:
        # Convert the chunk (which is a list of lists) into a string
        chunk_str = "\n".join([item[0] for item in chunk])

        if i-2 >= 0 and i-2 < len(chunk_context):
            context_chunk_str = "\n".join([item[0] for item in chunk_context[i-2]])
        else:
            context_chunk_str = chunk_str

        start_time = chunk[0][1]
        end_time = chunk[-1][2]

        results = extract_qa(chunk_str)  # let's assume it returns a list of dictionaries

        new_pairs = []


        for pair in results:
            # If this is a repeated question, ignore it and any answers that follow
            if pair.get("Q") == last_pair.get("Q"):
                if pair.get("A") == last_pair.get("A"):
                    continue
                last_pair["A"] = pair.get("A")
                continue
                

            # Otherwise, remember the question and keep the pair
            last_pair = pair
            pair["prompt_data"] = chunk_str
            pair["context"] = context_chunk_str
            pair["start"] = start_time
            pair["end"] = end_time

            new_pairs.append(pair)

        qa_pairs.extend(new_pairs)

        with open(path.split('.')[0] + "_analysis.json", "w") as fd:
            fd.write(json.dumps(qa_pairs, indent=2))
        
        i+=1

    return qa_pairs


def extract_qa(data):
    
    # examples to prime the model
    few_shot_ex="Extract the question and answer from the following transcripts as a [{\"Q\":\"_\"},{\"A\":\"_\"}], if there is no question or answer, return []:\n\n\"\"\"\nbut it's two separate, like, spools?\nyeah, it's two separate spools.\nso, okay.\n\"\"\"\nThere is one question and answer\n[{\"Q\":\"but it's two separate, like, spools?\",\"A\":\"yeah, it's two separate spools.\"}]\n\n\"\"\"\nso the first thing you do is, when you, let's say we want, let's, I don't know, let's grab a different color, just for fun.\nwhat color would you like?\npurple.\npurple? this one?\nyeah.\nsounds good.\n\"\"\"\nThere are multiple questions and answers\n[{\"Q\":\"what color would you like?\",\"A\":\"purple.\"},{\"Q\":\"purple? this one?\",\"A\":\"yeah.\"}]\n\n\"\"\"\num, so, when you have a spool, you might be able to go into here and find a bobbin that matches the color.\num, these are called bobbins, by the way.\n\"\"\"\nNo question.\n[]\n\n\"\"\"\nso, you can take this, and you put it on the end of the spool, with the little...\ndo i need to take the string off?\nno, no, no, that's fine.\nother way.\nthis way?\nperfect.\nyeah, so then, what you want to do is, you want to take this and push it to the right.\n\"\"\"\nThere are multiple questions and answers\n[{\"Q\":\"do i need to take the string off?\",\"A\":\"no, no, no, that's fine.\"},{\"Q\":\"this way?\",\"A\":\"perfect.\"}]\n\n\"\"\"\noh, you need to turn it on first.\nso then, it's on as this one?\nmhm.\nand then, hold this little guy right here.\n\"\"\"\nThere is one question and answer\n[{\"Q\":\"so then, it's on as this one?\",\"A\":\"mhm.\"}]\n\n\"\"\"\nbecause, I think, because it's like a cute name, they're bobbins, and they're small.\nI don't know.\nyeah.\nso, um, it looks like that's probably the, no, that's a different color.\n\"\"\"\nNo question.\n[]\n"
    
    response = completion_with_backoff(
    model="text-davinci-003",
    prompt=few_shot_ex + "\n\"\"\"" + data + "\n\"\"\"\n",
    temperature=0,
    max_tokens=64,
    top_p=1,
    best_of=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=["]"]
    )

    s = response.choices[0].text + "]"
    


    matches = re.findall(r'\[{.*?}\]', s)
    for match in matches:
        try:
            # Parse the fixed string as JSON
            out = json.loads(match)
            if out == None:
                return []
            return out
        except json.JSONDecodeError:
            return []
    return []


def main(transcript_path):

    transcript = []
    with open(transcript_path, "r") as f:
        transcript = json.loads(f.read())

    #Main
    qa_pairs = sliding_window(transcript, 4, 3, path=transcript_path)


    with open(transcript_path.split('.')[0] + "_analysis.json", "w") as fd:
        fd.write(json.dumps(qa_pairs, indent=2))

    print(qa_pairs)

if __name__ == "__main__":
    main(sys.argv[1])
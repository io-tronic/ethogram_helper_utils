#!/usr/bin/env python

import os
import sys
import json
import webvtt
from rapidfuzz import fuzz
from rapidfuzz import process
from datetime import datetime
import csv



def fuzzy_find_eval(query, str_list):
    """
    Rank a list of strings by their similarity to a query string.
    The list is returned in descending order of similarity.
    """
    # Create a list to store scores
    scores = []

    # For each string in the list
    for s in str_list:
        # Get all substrings
        substrings = [s[i: j] for i in range(len(s)) for j in range(i + 1, len(s) + 1)]
        # Find best match among substrings
        best_match, score, index = process.extractOne(query, substrings)
        # Append to scores list
        scores.append(score)
        
    
    return scores


def get_time(timestamp):
    return datetime.strptime(timestamp, "%H:%M:%S.%f")
        

def timestamp(ethogram,vtt):

    for analysis in ethogram:
        
        prompt_lines = analysis["prompt_data"].split('\n')

        caption_numbers = []
        
        #iterate through captions looking for start time
        for i in range(len(vtt)):
            # if the caption time is equal to the start of the context, add the numbers representing the caption
            if vtt[i].start == analysis["start"]:
                caption_numbers.extend(range(i, min(i+len(prompt_lines), len(vtt))))
                print("prompt found @ " + vtt[caption_numbers[0]].start)
                break
        else:  # This is executed if the for loop completes without finding a match
            print(f"No matching prompt found for start time {analysis['start']}")
            continue  # Skip to next iteration of the outer loop
        
        caption_text = [vtt[i].text for i in caption_numbers]
        
        #find fuzzy scores
        q_scores = fuzzy_find_eval( analysis["Q"], caption_text)
        a_scores = fuzzy_find_eval(analysis["A"], caption_text)
        print(analysis["Q"])
        print(caption_numbers)
        print(q_scores)

        if len(caption_numbers) < len(q_scores) or len(caption_numbers) < len(a_scores):
            raise ValueError("The length of 'caption_numbers' is less than the length of 'q_scores' or 'a_scores'")

        #find best match for q
        max_score = 0
        for i, score in enumerate(q_scores):
            if score >= max_score:
                max_score = score
                analysis["q_start"] = vtt[caption_numbers[i]].start
                analysis["q_end"] = vtt[caption_numbers[i]].end

        #find best match for a
        max_score = 0
        for i, score in enumerate(a_scores):
            if score >= max_score:
                max_score = score
                analysis["a_start"] = vtt[caption_numbers[i]].start
                analysis["a_end"] = vtt[caption_numbers[i]].end


        #start and end o qa sequence
        analysis["qa_start"] = analysis["q_start"]
        analysis["qa_end"] = analysis["a_end"]

        if get_time(analysis["qa_start"]) > get_time(analysis["qa_end"]):
            analysis["class"] = "unk-4"
            a = analysis["qa_start"]
            analysis["qa_start"] = analysis["qa_end"]
            analysis["qa_end"] = a
            print("malformed timestamp @ ")
    

    return ethogram

def main(ethogram_path, vtt_path):
    
    #Load Files
    vtt = webvtt.read(vtt_path)
    ethogram = None
    with open(ethogram_path, 'r') as f:
        ethogram = json.loads(f.read())

    timestamp(ethogram,vtt)
    
    #Save information
    new_filepath = ethogram_path.split('.')[0].split('_')[0]
    with open(new_filepath + "_ethogram_timestamp.json", "w") as fd:
        fd.write(json.dumps(ethogram, indent=2))


    # specify the fields for csv file
    fields = ethogram[0].keys()

    with open(new_filepath + "_ethogram_timestamp.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ethogram:
            writer.writerow(row)
        



if __name__ == "__main__":
    main(sys.argv[1],sys.argv[2])
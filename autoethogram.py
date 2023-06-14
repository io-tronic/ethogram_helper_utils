#!/usr/bin/env python

import os
import sys
import openai
import webvtt
from datetime import datetime, timedelta
import json
import csv

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

openai.api_key = os.getenv("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def get_category_inference(data):

    system_message = """
You are classifying these question/answer dialog interactions into four categories:

### 1. *Knowledge*

These are questions aimed at understanding what the other person already knows or has understood about a topic or a situation. These questions can often be seen at the beginning of a conversation or when introducing a new topic. They help the speaker gauge the existing knowledge of the listener.

For example:

- You've heard about this before, right?
- You know how to turn on the air conditioning, don't you?
- so, do you have any idea how this works?

### 2. *Clarification*

These questions are asked when the speaker is unsure about something that has just been said or explained. They are seeking a clearer understanding or want to make sure they've understood correctly. Repetition, reformulation, and confirmation questions are common types within this category.

For example:

- Hold on, what did you say was the first step again?
- what is that called?
- so it's this part?



### 3. *Feedback*

Identify questions that seek to validate or confirm the questioner's actions, decisions, or understanding. These questions are often asked after the participant has taken an action or given a response. They aim to assess the appropriateness or correctness of what has been done, thus providing an opportunity for corrective feedback.

For example:

- Am I taking the right approach to solve this problem?
- Did I complete this task correctly?
- Based on what you've seen, am I on the right track?

### 4. *Future*

Identify questions that ask for guidance, advice, or direction about what to do next or in the future. These questions are asked when the participant is unsure about what to do next or how to avoid a problem or mistake in the future. They seek proactive guidance and are often focused on future actions or plans.

For example:

- What would be the next best step in this process?
- How can I approach this task more effectively in the future?
- What should I keep in mind for similar tasks moving forward?

To better classify questions, pay attention to the context in which they are asked, the words and phrases used, and the type of information that is being sought. Please remember that categorizing questions might not always be clear-cut, and some questions could fall into more than one category. When in doubt, consider the primary intent of the question and the type of response it's likely to elicit.

Sure, here's how you might describe and expand upon an "Unknown/Other" category:

### 5. Unknown/Other - *Unknown*

Identify questions that do not fit neatly into the above categories. These may be questions that have ambiguous intent, are unclear, or are not oriented towards the topics of knowledge, clarification, feedback, or future actions. They could also be questions with multiple overlapping intents that cannot be classified into one category.

The "Unknown/Other" category acts as a catch-all for those questions that defy easy categorization. Remember, this doesn't mean these questions are unimportant or irrelevant, just that their classification is less straightforward than others.

For example:

- What's your favorite color?
- Did you watch the game last night?
- Why did the chicken cross the road?

These questions might be conversation starters, rhetorical questions, or simply casual exchanges unrelated to the research study's main focus. While coding, use this category for anything that doesn't fit into the other defined categories, or if you are in doubt about which category it falls into.
    """

    user_message_sep = "\n\n Given the above context, explain what category this interaction falls into? \n"

    user_message = data.get("context") + user_message_sep + '"Q": ' + data.get("Q") + "\n" + '"A": ' + data.get("A") 

    print(user_message)

    response = chat_completion_with_backoff(
    # response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
    temperature = 0.2
    )

    response_text = response.choices[0].message.content
    print(response_text)
    data["response"] = response_text
    data["prompt"] = user_message

    classification = None
    count = 0

    if "knowledge" in response_text.lower():
        count+=1
        classification = "knowledge"
    if "clarification" in response_text.lower():
        count+=1
        classification = "clarification"
    if "feedback" in response_text.lower():
        count+=1
        classification = "feedback"
    if "future" in response_text.lower():
        count+=1
        classification = "future"

    # If classification not assigned, or assigned too many times, return unk
    if "future" in response_text.lower():
        classification = "unk-3"
        data["class"] = classification
        return classification
    if count == 0:
        classification = "unk-1"
        data["class"] = classification
        return classification
    if count > 1:
        classification = "unk-2"
        data["class"] = classification
        return classification

    data["class"] = classification
    return classification


def main(transcript_analysis_path):

    new_filepath = transcript_analysis_path.split('.')[0].split('_')[0]

    analysis = []
    with open(transcript_analysis_path, "r") as f:
        analysis = json.loads(f.read())

    for q in analysis:
        classification = get_category_inference(q)
        print(classification)
        print("----------\n\n")
        with open(new_filepath + "_ethogram.json", "w") as fd:
            fd.write(json.dumps(analysis, indent=2))


    

    with open(new_filepath + "_ethogram.json", "w") as fd:
        fd.write(json.dumps(analysis, indent=2))


    # specify the fields for csv file
    fields = analysis[0].keys()

    # write the data into a CSV file
    with open(new_filepath + "_ethogram.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in analysis:
            writer.writerow(row)

    print(analysis)

if __name__ == "__main__":
    main(sys.argv[1])
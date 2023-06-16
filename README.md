This repository contains utilities used for our final project in COGS13 with Professor Rossano. It provides a pipeline for processing audio into transcripts, identifying the questions and answers in those transcripts, classifying those questions, and timestamping those questions. Useful in combination with tools like ELAN and as an assistant to human coders.

Everything is based on non-local AI categorization and as such all the usual concerns and biases about that that apply. Requires an OpenAI API key.

### 1. `./gettranscript.py 'audio file path'` 
- Returns transcript as .json, .txt, and .vtt
- Uses OpenAI Whisper to transcribe 
- Segments audio into 120s chunks and uses mixed prompt to maintain consistent style.
  - tokens from last transcript mixed with style reference

### 2. `./id_questions.py 'transcript json path'`
- Uses OpenAI text-davinci-003 model on scrolling window to extract questions.
- This function is very expensive. For large datasets, training a less expensive classifier would be highly recommended. No model smaller than text-davinci-003 worked in our (untrained) use case.
- high false positive rate

### 3. `./autoethogram.py 'transcript analysis path' `
- Uses gpt-35-turbo as a classifier to classify question type
- Low accuracy in our use case, YMMV
- Cheap

### 4. `./timestamper.py 'ethogram path' 'vtt transcript path' `
- fuzzy matches questions found by `./id_questions` with captions in the context window in order to timestamp the question, response, and interaction with phrase-level accuracy (dependent on performance of classifier)

---

Written by Vittoria Tenino for a project with Allison Chan and Caroline Jiang.

---

```
Copyright (c) 2023 Vittoria Tenino

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

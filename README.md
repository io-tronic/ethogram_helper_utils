This repository contains utilities used for our final project in COGS13 with Professor Rossano

### `./gettranscript.py 'audio file path'` 
- Returns transcript as .json, .txt, and .vtt
- Uses OpenAI Whisper to transcribe 
- Segments audio into 120s chunks and uses mixed prompt to maintain consistent style.
  - tokens from last transcript mixed with style reference

### `./id_questions.py 'transcript json path'`
- Uses OpenAI text-davinci-003 model on scrolling window to extract questions.
- This function is very expensive. For large datasets, training a less expensive classifier would be highly recommended. No model smaller than text-davinci-003 worked in our use case
- high false positive rate

### `./autoethogram.py 'transcript analysis path'
- Uses gpt-35-turbo as a classifier to classify question type
- Low accuracy in our use case, YMMV
- Cheap

### `./timestamper.py 'ethogram path' 'vtt transcript path' `
- fuzzy matches questions found by `./id_questions` with captions in the context window in order to timestamp the question, response, and interaction with phrase-level accuracy (dependent on performance of classifier)
# 📜 Meeting Minutes Generator

## Overview
This is a Streamlit-based web application that processes audio recordings of meetings and generates structured minutes. The app transcribes speech using OpenAI's Whisper model, performs speaker diarization with Gemini AI, summarizes key points, analyzes sentiment, extracts keywords, and detects action items. The final minutes can be downloaded as a Word document.

## Features
- **Speech-to-Text Transcription**: Converts audio recordings into text using Whisper.
- **Speaker Diarization**: Identifies different speakers using the Gemini AI model.
- **Summarization**: Generates a concise summary of the meeting.
- **Sentiment Analysis**: Classifies the meeting tone as Positive, Negative, or Neutral.
- **Keyword Extraction**: Identifies the most relevant keywords from the meeting.
- **Action Item Detection**: Extracts actionable points from the discussion.
- **Download as Word Document**: Saves the meeting minutes as a `.docx` file.

## Prerequisites
### Install Dependencies
Ensure you have Python installed. Install required packages using:
```sh
pip install streamlit torch whisper google-generativeai python-docx keybert
```

### Set Up Gemini API Key
You need to set the `GEMINI_API_KEY` as an environment variable:
```sh
export GEMINI_API_KEY='your-api-key-here'  # Mac/Linux
set GEMINI_API_KEY=your-api-key-here  # Windows
```

## How to Run the App
1. Clone this repository:
   ```sh
   git clone https://github.com/your-repo/meeting-minutes-generator.git
   cd meeting-minutes-generator
   ```
2. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```
3. Upload an audio file (`.mp3`, `.wav`, `.m4a`) and generate structured meeting minutes.

## Usage
- Upload an audio file.
- The app will process the file and display:
  - Transcribed text
  - Speaker-labeled conversation
  - Summary
  - Sentiment analysis
  - Extracted keywords
  - Action items
- Download the results as a Word document.

## Technologies Used
- **Streamlit**: UI for web-based interaction.
- **Whisper**: Speech-to-text transcription.
- **Google Gemini AI**: Speaker diarization, summarization, sentiment analysis, and action item extraction.
- **KeyBERT**: Keyword extraction.
- **Python-docx**: Creating downloadable meeting minutes.






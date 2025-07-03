# app.py
from flask import Flask, render_template, request, jsonify
from youtube_transcript_api import YouTubeTranscriptApi as YTapi
from youtube_transcript_api._errors import VideoUnavailable
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from urllib.parse import urlparse, parse_qs

app = Flask(__name__)

# Load the model and tokenizer only once
checkpoint = "sshleifer/distilbart-cnn-12-6"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint)

# Utility to extract video ID from YouTube URL
def extract_video_id(url):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:]
    elif query.hostname in ('www.youtube.com', 'youtube.com'):
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
    return None

# Summarization function
def summarize(text):
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"])
    summary = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return summary[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    data = request.form
    video_url = data.get('video_url')
    video_id = extract_video_id(video_url)

    if not video_id:
        return "Invalid YouTube URL. Please provide a valid link.", 400

    try:
        transcript = YTapi.get_transcript(video_id)
    except VideoUnavailable:
        return render_template('result.html', transcript="N/A", summary="Transcript is not available for this video.")

    transcript_text = ' '.join([i['text'] for i in transcript])
    summary = summarize(transcript_text)

    return render_template('result.html', transcript=transcript_text, summary=summary)

if __name__ == '__main__':
    app.run(debug=True)

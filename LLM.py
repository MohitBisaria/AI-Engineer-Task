import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from flask import Flask, request, jsonify

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Flask app
app = Flask(__name__)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# Create a summarization pipeline
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)


@app.route("/summarise", methods=["GET", "POST"])
def summarise():
    """
    Endpoint to summarize the input text.
    """
    # Handle GET requests
    if request.method == "GET":
        prompt = request.args.get("prompt")
        if prompt:
            # Summarize the text
            summary = summarizer(prompt, max_length=100, min_length=30)[0]["summary_text"]
            return jsonify({"response": summary})
        else:
            return jsonify({"error": "Missing 'prompt' parameter"}), 400

    # Handle POST requests
    elif request.method == "POST":
        data = request.get_json()
        prompt = data.get("prompt")
        if prompt:
            # Summarize the text
            summary = summarizer(prompt, max_length=100, min_length=30)[0]["summary_text"]
            return jsonify({"response": summary})
        else:
            return jsonify({"error": "Missing 'prompt' in request body"}), 400

    # If the request method is neither GET nor POST, return an error
    else:
        return jsonify({"error": "Invalid request method"}), 405 

if __name__ == "__main__":
        app.run(debug=True)
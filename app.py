from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import difflib

app = Flask(__name__)
CORS(app)  # allow requests from any origin, adjust in production if needed

# Load Whisper model once at startup
model = whisper.load_model("base")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "audio" not in request.files or "text" not in request.form:
        return jsonify({"error": "Missing audio or text input"}), 400

    audio_file = request.files["audio"]
    target_text = request.form["text"].strip().lower()

    # Save upload temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        audio_file.save(temp_audio.name)
        result = model.transcribe(temp_audio.name)
        os.unlink(temp_audio.name)

    spoken_text = result["text"].strip().lower()
    spoken_words = spoken_text.split()
    target_words = target_text.split()

    # Word-level similarity score
    matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
    score = round(matcher.ratio() * 100, 2)

    # Word-level feedback
    feedback = [
        {"word": word, "status": "correct" if word in spoken_words else "missed"}
        for word in target_words
    ]

    return jsonify({
        "transcript": spoken_text,
        "score": score,
        "feedback": feedback
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

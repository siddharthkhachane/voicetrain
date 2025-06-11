from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os
import difflib
import shutil

app = Flask(__name__)
CORS(app, resources={r"/transcribe": {"origins": "*"}})

if not shutil.which("ffmpeg"):
    raise EnvironmentError("ffmpeg not found. Please install it and add to PATH.")

model = whisper.load_model("base")

@app.route("/transcribe", methods=["POST"])
def transcribe():
    try:
        if "audio" not in request.files or "text" not in request.form:
            return jsonify({"error": "Missing audio or text input"}), 400

        audio_file = request.files["audio"]
        target_text = request.form["text"].strip().lower()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
            audio_file.save(temp_audio.name)
            result = model.transcribe(temp_audio.name)
            os.unlink(temp_audio.name)

        spoken_text = result["text"].strip().lower()
        spoken_words = spoken_text.split()
        target_words = target_text.split()

        matcher = difflib.SequenceMatcher(None, target_words, spoken_words)
        score = round(matcher.ratio() * 100, 2)

        feedback = [
            {"word": word, "status": "correct" if word in spoken_words else "missed"}
            for word in target_words
        ]

        return jsonify({
            "transcript": spoken_text,
            "score": score,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

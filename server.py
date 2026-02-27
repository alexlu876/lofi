from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import subprocess
import tempfile
import os
import re

app = Flask(__name__)
CORS(app, supports_credentials=False, resources={r"/*": {"origins": "*"}})

YOUTUBE_URL_RE = re.compile(
    r"^https?://(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)"
)


@app.route("/fetch", methods=["POST"])
def fetch_audio():
    data = request.get_json(silent=True)
    if not data or not data.get("url"):
        return jsonify({"error": "Missing url"}), 400

    url = data["url"]
    if not YOUTUBE_URL_RE.match(url):
        return jsonify({"error": "Invalid YouTube URL"}), 400

    tmpdir = tempfile.mkdtemp()
    output_template = os.path.join(tmpdir, "audio.%(ext)s")

    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "bestaudio",
                "--no-playlist",
                "-o", output_template,
                url,
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print("yt-dlp stderr:", result.stderr)
            print("yt-dlp stdout:", result.stdout)
            return jsonify({"error": "yt-dlp failed", "details": result.stderr}), 500

        # Find whatever audio file yt-dlp wrote (webm, opus, m4a, mp3, etc.)
        audio_path = None
        for f in os.listdir(tmpdir):
            audio_path = os.path.join(tmpdir, f)
            break

        if not audio_path:
            return jsonify({"error": "No audio file found after download"}), 500

        ext = os.path.splitext(audio_path)[1].lower()
        mimetypes = {
            ".webm": "audio/webm",
            ".opus": "audio/opus",
            ".m4a": "audio/mp4",
            ".mp3": "audio/mpeg",
            ".ogg": "audio/ogg",
        }
        mimetype = mimetypes.get(ext, "application/octet-stream")

        response = send_file(audio_path, mimetype=mimetype)

        @response.call_on_close
        def cleanup():
            for f in os.listdir(tmpdir):
                os.remove(os.path.join(tmpdir, f))
            os.rmdir(tmpdir)

        return response

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Download timed out"}), 504
    except FileNotFoundError:
        return jsonify({"error": "yt-dlp not found. Install it with: pip install yt-dlp"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("Lofi server running on http://localhost:5001")
    app.run(port=5001, debug=True)

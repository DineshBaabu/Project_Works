from flask import Flask, request, render_template
import subprocess

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form["video_url"]
        # extract the transcript from the video
        transcript = extract_transcript(video_url)
        return render_template("index.html", transcript=transcript)
    return render_template("index.html")

def extract_transcript(video_url):
    # implementation to extract the transcript from the video
    output = subprocess.run("whisper hehe.mp3", capture_output=True, text=True, shell=True)
    transcript = output.stdout
    return transcript

if __name__ == "__main__":
    app.run()

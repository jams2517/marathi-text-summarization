from flask import Flask, render_template, request
from preprocess import clean_text
from textrank import text_rank_abstractive
import io

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8")
            # Generate abstractive summary
            summary = text_rank_abstractive(file_content)
            return render_template("index.html", original_text=file_content, summary=summary)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

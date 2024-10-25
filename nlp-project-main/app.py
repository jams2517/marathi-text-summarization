from flask import Flask, render_template, request
from preprocess import clean_text
from textrank import text_rank_abstractive
import io

app = Flask(__name__)

def calculate_word_overlap(original, summary):
    original_words = set(original.split())
    summary_words = set(summary.split())
    overlap = original_words.intersection(summary_words)
    return len(overlap), len(original_words), len(summary_words)

def calculate_precision_recall(original, summary):
    overlap_count, original_count, summary_count = calculate_word_overlap(original, summary)
    
    precision = overlap_count / summary_count if summary_count > 0 else 0
    recall = overlap_count / original_count if original_count > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score

def calculate_weighted_accuracy(original, summary):
    original_words = original.split()
    summary_words = summary.split()

    # Count occurrences of each word in the original
    original_word_count = {word: original_words.count(word) for word in set(original_words)}
    overlap_count = sum(original_word_count[word] for word in summary_words if word in original_word_count)

    total_weight = sum(original_word_count.values())

    accuracy = overlap_count / total_weight if total_weight > 0 else 0
    return accuracy


@app.route("/", methods=["GET", "POST"])
def summarize():
    if request.method == "POST":
        uploaded_file = request.files.get("file")
        if uploaded_file:
            file_content = uploaded_file.read().decode("utf-8")
            # Generate abstractive summary
            summary = text_rank_abstractive(file_content)

            # Calculate precision, recall, F1 score, and accuracy
            precision, recall, f1_score = calculate_precision_recall(file_content, summary)
            accuracy = calculate_weighted_accuracy(file_content, summary)

            # Print performance metrics to the terminal
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1_score:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

            return render_template("index.html", original_text=file_content, summary=summary)
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

import os
import pickle
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

try:
    import docx
except Exception:
    docx = None

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"txt", "pdf", "docx"}
MODEL_PATH = os.path.join("models", "career_model.pkl")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = "change-this-secret"

# Skills vocabulary (lowercase)
SKILLS_VOCAB = [s.lower() for s in [
    "Python", "Pandas", "NumPy", "SQL", "Java", "C++", "C#", "Unity", "Blender",
    "Flask", "Django", "HTML", "CSS", "JavaScript", "React", "Node.js", "Machine Learning",
    "Deep Learning", "TensorFlow", "PyTorch", "Tableau", "Power BI", "AWS", "GCP", "Kubernetes",
    "Docker", "Tensorflow", "PyTorch"
]]

model_pipeline = None
model_loaded = False
if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, "rb") as f:
            model_pipeline = pickle.load(f)
        model_loaded = True
        print("Loaded model from", MODEL_PATH)
    except Exception as e:
        print("Error loading model:", e)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(filepath):
    ext = filepath.rsplit(".", 1)[1].lower()
    try:
        if ext == "txt":
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == "pdf" and PyPDF2:
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        elif ext in ["docx", "doc"] and docx:
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
    except Exception as e:
        print("Error extracting text:", e)
        return ""

def extract_skills_rule_based(text):
    text_lower = text.lower()
    found = []
    for skill in SKILLS_VOCAB:
        if skill in text_lower and skill not in found:
            found.append(skill)
    return found

def predict_all_careers(text):
    """
    Returns a list of {"career": class_name, "prob": float_score} sorted desc.
    If model not available, returns None.
    """
    if not model_loaded or model_pipeline is None:
        return None
    try:
        probs = model_pipeline.predict_proba([text])[0]
        classes = list(model_pipeline.classes_)
        class_probs = [{"career": c, "prob": float(p)} for c, p in zip(classes, probs)]
        sorted_list = sorted(class_probs, key=lambda x: x["prob"], reverse=True)
        return sorted_list
    except Exception as e:
        print("Model prediction error:", e)
        return None

@app.route("/")
def index():
    return render_template("index.html", model_loaded=model_loaded)

@app.route("/predict", methods=["POST"])
def predict():
    if "resume" not in request.files:
        flash("No file part named 'resume' in request.")
        return redirect(url_for("index"))

    file = request.files["resume"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Unsupported file type.")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    text = extract_text_from_file(path)
    
    try:
        os.remove(path)
    except Exception:
        pass

    if not text.strip():
        return render_template("results.html",
                               error="Could not extract text from the uploaded file.",
                               model_loaded=model_loaded)

    skills = extract_skills_rule_based(text)
    career_scores = predict_all_careers(text) if model_loaded else None

    top_pred = career_scores[0] if career_scores else None

    if career_scores:
        for c in career_scores:
            c["percent"] = round(c["prob"] * 100, 2)

    return render_template("results.html",
                           text_snippet=text[:4000],
                           skills=skills,
                           career_scores=career_scores,
                           top_pred=top_pred,
                           model_loaded=model_loaded)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

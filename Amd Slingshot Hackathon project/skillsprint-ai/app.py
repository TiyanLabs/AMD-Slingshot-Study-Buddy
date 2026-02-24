import os
import pandas as pd
import joblib
from flask import Flask, render_template, request, redirect, url_for, session
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Load Environment Variables
# -----------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# -----------------------------
# App Configuration
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "super_secret_key_change_this"

# -----------------------------
# Load ML Model & Encoders
# -----------------------------
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
topic_encoder = joblib.load(os.path.join(BASE_DIR, "topic_encoder.pkl"))
strength_encoder = joblib.load(os.path.join(BASE_DIR, "strength_encoder.pkl"))

# -----------------------------
# Dummy Users (Replace with DB later)
# -----------------------------
users = {
    "admin": "1234",
    "student": "abcd"
}

# -----------------------------
# Login Route
# -----------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        if username in users and users[username] == password:
            session["user"] = username
            return redirect(url_for("home"))
        else:
            return render_template("login.html", error="Invalid credentials")

    return render_template("login.html")


# -----------------------------
# Logout
# -----------------------------
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("login"))


# -----------------------------
# Home (Protected)
# -----------------------------
@app.route("/")
def home():
    if "user" not in session:
        return redirect(url_for("login"))

    return render_template("index.html", user=session["user"])


# -----------------------------
# Prediction Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "user" not in session:
        return redirect(url_for("login"))

    topic = request.form["topic"]
    quiz_no = int(request.form["quiz_no"])
    time_taken = int(request.form["time_taken"])

    # Encode topic
    topic_encoded = topic_encoder.transform([topic])[0]

    # Model Prediction
    input_data = [[quiz_no, time_taken, topic_encoded]]
    prediction = model.predict(input_data)
    strength = strength_encoder.inverse_transform(prediction)[0]

    # ------------------------------------
    # AI Generated Study Plan (LLM)
    # ------------------------------------
    try:
        prompt = f"""
        You are an expert academic performance coach.

        The student is currently classified as: {strength}
        Subject: {topic}
        Average quiz time: {time_taken} minutes.

        Generate:
        1. Personalized improvement strategy
        2. Weekly study plan
        3. Revision technique
        4. Practice recommendation

        Make it structured and motivating.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        study_plan = response.choices[0].message.content

    except Exception as e:
        study_plan = "AI Study Plan could not be generated."

    # ------------------------------------
    # Real Historical Data from CSV
    # ------------------------------------
    df = pd.read_csv(os.path.join(BASE_DIR, "student_data.csv"))

    student_history = df[df["topic"] == topic].head(7)

    performance_data = student_history["score"].tolist()
    labels = student_history["quiz_no"].tolist()

    # Badge Color Logic
    if strength == "Weak":
        badge_color = "danger"
    elif strength == "Moderate":
        badge_color = "warning"
    else:
        badge_color = "success"

    return render_template(
        "result.html",
        strength=strength,
        study_plan=study_plan,
        badge_color=badge_color,
        performance_data=performance_data,
        labels=labels,
        user=session["user"]
    )


# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)

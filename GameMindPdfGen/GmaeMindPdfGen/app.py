from flask import Flask, send_file
import requests
import pandas as pd
from utils import conversion, random_forest_train

app = Flask(__name__)
firebase_url = "https://my-application-91b91-default-rtdb.firebaseio.com/"


@app.route("/")
def home():
    return "Home"


@app.route("/pdf/<username>")
def pdfGenerator(username):
    response = requests.get(firebase_url + ".json")
    if response.status_code == 200:
        data = response.json()
        user_data = data.get("user", None).get(username, None)
        if user_data is not None:
            rows_game1 = []
            for user, user_data in data.get("user", {}).items():
                age = user_data.get("age")
                game1_sessions = user_data.get("sessions")
                if isinstance(game1_sessions, dict):
                    for session, session_data in game1_sessions.items():
                        if isinstance(session_data, dict):
                            flat_data = {"user": user, "age": age, "session": session}
                            flat_data.update(session_data)
                            rows_game1.append(flat_data)
            corrected_df = conversion.normalize_and_correct(pd.DataFrame(rows_game1))
            pdf_output = random_forest_train.generate_cognitive_report(
                username, corrected_df
            )
            return send_file(
                pdf_output,
                as_attachment=True,
                download_name=f"{username}_Cognitive_Report.pdf",
                mimetype="application/pdf",
            )

    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "ERROR"


@app.route("/train")
def train():
    response = requests.get(firebase_url + ".json")
    if response.status_code == 200:
        data = response.json()
        rows_game1 = []
        for user, user_data in data.get("user", {}).items():
            age = user_data.get("age")
            game1_sessions = user_data.get("sessions")
            if isinstance(game1_sessions, dict):
                for session, session_data in game1_sessions.items():
                    if isinstance(session_data, dict):
                        flat_data = {"user": user, "age": age, "session": session}
                        flat_data.update(session_data)
                        rows_game1.append(flat_data)
        corrected_df = conversion.normalize_and_correct(pd.DataFrame(rows_game1))
        random_forest_train.random_forest_train(corrected_df)
        return "Trained", 200

    else:
        return "Unable to fetch", 400


if __name__ == "__main__":
    app.run(port=8000, host="0.0.0.0")

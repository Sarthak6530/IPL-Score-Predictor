from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load("forest_model.pkl")

teams = [
    'Chennai Super Kings',
    'Delhi Daredevils',
    'Kings XI Punjab',
    'Kolkata Knight Riders',
    'Mumbai Indians',
    'Rajasthan Royals',
    'Royal Challengers Bangalore',
    'Sunrisers Hyderabad'
]

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        batting_team = request.form["batting_team"]
        bowling_team = request.form["bowling_team"]
        runs = int(request.form["runs"])
        wickets = int(request.form["wickets"])
        overs = float(request.form["overs"])
        runs_last_5 = int(request.form["runs_last_5"])
        wickets_last_5 = int(request.form["wickets_last_5"])

        input_data = []

        for team in teams:
            input_data.append(1 if team == batting_team else 0)

        for team in teams:
            input_data.append(1 if team == bowling_team else 0)

        input_data.extend([runs, wickets, overs, runs_last_5, wickets_last_5])

        final_input = np.array([input_data])
        prediction = int(round(model.predict(final_input)[0]))

    return render_template("index.html", teams=teams, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scalers
model = pickle.load(open("model.pkl", "rb"))
ms = pickle.load(open("minmaxscaler.pkl", "rb"))
sc = pickle.load(open("standscaler.pkl", "rb"))

# Crop dictionary mapping index to crop name
crop_dict = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans", 
    21: "Chickpea", 22: "Coffee"
}

@app.route("/")
def home():
    return render_template("index.html", crops=None)  # Ensure no results are sent

@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    N = float(request.form["Nitrogen"])
    P = float(request.form["Phosporus"])
    K = float(request.form["Potassium"])
    temp = float(request.form["Temperature"])
    humidity = float(request.form["Humidity"])
    ph = float(request.form["Ph"])
    rainfall = float(request.form["Rainfall"])

    # Create feature array and scale
    feature_list = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    scaled_features = ms.transform(feature_list)
    final_features = sc.transform(scaled_features)

    # Get probabilities of all crops
    probabilities = model.predict_proba(final_features)[0]

    # Get top 3 crop indices
    top_3_indices = np.argsort(probabilities)[-3:][::-1]

    # Define rankings and messages
    rankings = ["Top 1 Crop", "Top 2 Crop", "Top 3 Crop"]
    recommendation_messages = [
        "Highly suitable for cultivation!",
        "Good choice for this soil and climate.",
        "Recommended based on analysis."
    ]

    # Prepare top 3 crops with ranking
    top_3_crops = [
        {
            "rank": rankings[idx],  
            "name": crop_dict[i + 1],  
            "message": recommendation_messages[idx]
        }
        for idx, i in enumerate(top_3_indices)
    ]

    return render_template("index.html", crops=top_3_crops)

@app.route("/reset")
def reset():
    return redirect(url_for("home"))  # Redirect to clear results

if __name__ == "__main__":
    app.run(debug=True)

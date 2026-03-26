from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('best_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    screen_time = float(request.form['daily_screen_time'])
    social = float(request.form['social_media'])
    gaming = float(request.form['gaming'])
    work = float(request.form['work_study'])
    sleep_hours = float(request.form['sleep_hours'])
    stress_level = int(request.form['stress_level']) # Kinuha natin ang stress level
    
    other_time = max(0, screen_time - (social + gaming + work))
    
    features = [
        float(request.form['age']), int(request.form['gender']),
        screen_time, social, gaming, work, sleep_hours,
        int(request.form['notifications']), int(request.form['app_opens']),
        float(request.form['weekend_screen_time']), stress_level,
        int(request.form['academic_impact'])
    ]
    
    prediction = model.predict([features])[0]
    raw_prob = model.predict_proba([features])[0][1] 
    confidence_score = raw_prob * 100 
    
    result = "Addicted" if prediction == 1 else "Not Addicted"
    
    # Ipinasa natin ang kumpletong data para sa 3 graphs
    user_data = {
        'daily_screen_time': screen_time, 
        'sleep_hours': sleep_hours,
        'stress_level': stress_level,
        'activity_data': [social, gaming, work, other_time] 
    }
    
    return render_template('result.html', 
                           prediction_text=result, 
                           confidence=round(confidence_score, 2), 
                           raw_prob=raw_prob, 
                           user_data=user_data)

if __name__ == "__main__":
    app.run(debug=True)
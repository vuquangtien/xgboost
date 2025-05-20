from flask import Flask, request, jsonify
from googleapiclient.discovery import build
import re
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import isodate
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

API_KEY = "AIzaSyDwtQQ4KkipWnfT1YxCVSTfGhkH56Ga3p0"
youtube = build("youtube", "v3", developerKey=API_KEY)

model_bert = SentenceTransformer('all-MiniLM-L6-v2')
model = joblib.load('xgboost_model.pkl')
scaler_text = joblib.load('scaler_text_xgb.pkl')
scaler_category = joblib.load('scaler_category_xgb.pkl')
scaler_days = joblib.load('scaler_days_xgb.pkl')
scaler_duration = joblib.load('scaler_duration_xgb.pkl')

def get_video_id(youtube_url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
    if video_id:
        return video_id.group(1)
    else:
        raise ValueError("Could not find video ID in URL!")

def get_video_info(youtube_url):
    video_id = get_video_id(youtube_url)
    
    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    )
    response = request.execute()

    if not response["items"]:
        raise ValueError("No video found with this ID!")

    video_info = response["items"][0]["snippet"]
    content_details = response["items"][0]["contentDetails"]
    
    title = video_info.get("title", "")
    description = video_info.get("description", "")
    tags = video_info.get("tags", [])
    category_id = video_info.get("categoryId", "")
    
    category_request = youtube.videoCategories().list(
        part="snippet",
        id=category_id
    )
    category_response = category_request.execute()
    category = category_response["items"][0]["snippet"]["title"] if category_response["items"] else "Unknown"

    publish_date = video_info.get("publishedAt", "")
    
    publish_datetime = datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ")
    current_datetime = datetime.utcnow()
    days_since_publish = (current_datetime - publish_datetime).days
    
    duration_iso = content_details.get("duration", "PT0S")
    try:
        duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
    except:
        duration_seconds = 0

    return {
        "title": title,
        "description": description,
        "tags": tags,
        "category_id": int(category_id),
        "category": category,
        "publish_date": publish_date,
        "days_since_publish": int(days_since_publish),
        "video_durations": int(duration_seconds)
    }

@app.route('/analyze', methods=['POST'])
def analyze_video():
    data = request.json
    youtube_url = data.get('url', '')
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        video_info = get_video_info(youtube_url)
        
        text_combined = video_info["title"] + " " + video_info["description"]
        text_embedding = model_bert.encode([text_combined])[0]
        
        category_id = video_info["category_id"]
        days_to_trending = video_info["days_since_publish"]
        video_durations = video_info["video_durations"]
        
        text_embedding = scaler_text.transform([text_embedding])[0]
        category_id = scaler_category.transform([[category_id]])[0][0]
        days_to_trending = scaler_days.transform([[days_to_trending]])[0][0]
        video_durations = scaler_duration.transform([[video_durations]])[0][0]
        
        X_combined = np.hstack([
            np.array([category_id]),
            np.array([days_to_trending]),
            np.array([video_durations]),
            text_embedding
        ]).reshape(1, -1)
        
        prediction = model.predict_proba(X_combined)
        predicted_class = np.argmax(prediction, axis=1)[0]
        probabilities = prediction[0].tolist()
        
        labels = {0: "Not popular", 1: "Controversy", 2: "Decent", 3: "Overwhelming positive"}
        
        if predicted_class == 3:
            recommendation = "absolutely recommend"
        elif predicted_class in [1, 2]:
            recommendation = "recommend"
        else:
            recommendation = "not recommend"
        
        return jsonify({
            'video_info': {
                'title': video_info['title'],
                'category': video_info['category'],
                'publish_date': video_info['publish_date'],
                'days_since_publish': video_info['days_since_publish'],
                'duration': video_info['video_durations']
            },
            'prediction': {
                'class': int(predicted_class),
                'label': labels[predicted_class],
                'recommendation': recommendation,
                'probabilities': [{'label': labels[i], 'value': round(prob * 100, 2)} for i, prob in enumerate(probabilities)]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
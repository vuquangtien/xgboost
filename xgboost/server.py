from flask import Flask, request, jsonify
from googleapiclient.discovery import build
import re
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
import isodate
from datetime import datetime
from flask_cors import CORS
import logging
import json
import os
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("analysis_log.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

API_KEY = "AIzaSyDwtQQ4KkipWnfT1YxCVSTfGhkH56Ga3p0"
youtube = build("youtube", "v3", developerKey=API_KEY)

# Load models and log info about them
logger.info("Loading models and scalers...")

try:
    model_bert = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info(f"Successfully loaded SentenceTransformer model: all-MiniLM-L6-v2")
    
    # Check model file details
    model_files = [
        'xgboost_model.pkl',
        'scaler_text_xgb.pkl',
        'scaler_category_xgb.pkl',
        'scaler_days_xgb.pkl',
        'scaler_duration_xgb.pkl'
    ]
    
    for file_path in model_files:
        if os.path.exists(file_path):
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
            logger.info(f"File: {file_path}, Size: {file_size:.2f} MB, Last Modified: {mod_time}")
        else:
            logger.error(f"File does not exist: {file_path}")
    
    model_path = 'xgboost_model.pkl'
    model = joblib.load(model_path)
    logger.info(f"Successfully loaded main model from {model_path}")
    logger.info(f"Model type: {type(model).__name__}")
    
    # Load scalers
    scaler_text = joblib.load('scaler_text_xgb.pkl')
    logger.info(f"Loaded text scaler: {type(scaler_text).__name__}")
    
    scaler_category = joblib.load('scaler_category_xgb.pkl')
    logger.info(f"Loaded category scaler: {type(scaler_category).__name__}")
    
    scaler_days = joblib.load('scaler_days_xgb.pkl')
    logger.info(f"Loaded days scaler: {type(scaler_days).__name__}")
    
    scaler_duration = joblib.load('scaler_duration_xgb.pkl')
    logger.info(f"Loaded duration scaler: {type(scaler_duration).__name__}")
    
except Exception as e:
    logger.error(f"Error loading models: {str(e)}", exc_info=True)
    raise

def get_video_id(youtube_url):
    video_id = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11})", youtube_url)
    if video_id:
        return video_id.group(1)
    else:
        raise ValueError("Could not find video ID in URL!")

def get_video_info(youtube_url):
    video_id = get_video_id(youtube_url)
    logger.info(f"Extracted video ID: {video_id}")
    
    request = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=video_id
    )
    response = request.execute()
    
    if not response["items"]:
        logger.error(f"No video found with ID: {video_id}")
        raise ValueError("No video found with this ID!")

    logger.info(f"YouTube API response received. Items count: {len(response['items'])}")
    
    video_info = response["items"][0]["snippet"]
    content_details = response["items"][0]["contentDetails"]
    
    title = video_info.get("title", "")
    description = video_info.get("description", "")
    tags = video_info.get("tags", [])
    category_id = video_info.get("categoryId", "")
    
    logger.info(f"Video title: {title}")
    logger.info(f"Category ID: {category_id}")
    
    category_request = youtube.videoCategories().list(
        part="snippet",
        id=category_id
    )
    category_response = category_request.execute()
    category = category_response["items"][0]["snippet"]["title"] if category_response["items"] else "Unknown"
    
    logger.info(f"Category name: {category}")

    publish_date = video_info.get("publishedAt", "")
    
    publish_datetime = datetime.strptime(publish_date, "%Y-%m-%dT%H:%M:%SZ")
    current_datetime = datetime.utcnow()
    days_since_publish = (current_datetime - publish_datetime).days
    
    logger.info(f"Publish date: {publish_date}, Days since publish: {days_since_publish}")
    
    duration_iso = content_details.get("duration", "PT0S")
    try:
        duration_seconds = isodate.parse_duration(duration_iso).total_seconds()
    except:
        duration_seconds = 0
        
    logger.info(f"Duration ISO: {duration_iso}, Duration in seconds: {duration_seconds}")

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
    
    logger.info(f"Analyzing URL: {youtube_url}")
    
    if not youtube_url:
        return jsonify({'error': 'No URL provided'}), 400
    
    try:
        video_info = get_video_info(youtube_url)
        logger.info(f"Raw video info: {json.dumps(video_info, indent=2)}")
        
        text_combined = video_info["title"] + " " + video_info["description"]
        text_embedding = model_bert.encode([text_combined])[0]
        
        category_id = video_info["category_id"]
        days_to_trending = video_info["days_since_publish"]
        video_durations = video_info["video_durations"]
        
        # Log raw features before scaling
        logger.info(f"Raw features before scaling: category_id={category_id}, days_to_trending={days_to_trending}, video_durations={video_durations}")
        logger.info(f"Text embedding shape: {text_embedding.shape}")
        logger.info(f"Raw text embedding (first 10): {text_embedding[:10]}")
        
        # Store original values for comparison
        original_text_embedding = text_embedding.copy()
        original_category_id = category_id
        original_days_to_trending = days_to_trending
        original_video_durations = video_durations
        
        text_embedding = scaler_text.transform([text_embedding])[0]
        category_id = scaler_category.transform([[category_id]])[0][0]
        days_to_trending = scaler_days.transform([[days_to_trending]])[0][0]
        video_durations = scaler_duration.transform([[video_durations]])[0][0]
        
        # Log transformation details
        logger.info(f"TRANSFORMATION DETAILS:")
        logger.info(f"Text embedding: Original first 5: {original_text_embedding[:5]}")
        logger.info(f"Text embedding: Transformed first 5: {text_embedding[:5]}")
        logger.info(f"Category ID: Original={original_category_id}, Transformed={category_id}")
        logger.info(f"Days to trending: Original={original_days_to_trending}, Transformed={days_to_trending}")
        logger.info(f"Video duration: Original={original_video_durations}, Transformed={video_durations}")
        
        # Log scaled features
        logger.info(f"Scaled features: category_id={category_id}, days_to_trending={days_to_trending}, video_durations={video_durations}")
        
        X_combined = np.hstack([
            np.array([category_id]),
            np.array([days_to_trending]),
            np.array([video_durations]),
            text_embedding
        ]).reshape(1, -1)
        
        logger.info(f"Combined feature vector shape: {X_combined.shape}")
        logger.info(f"Feature vector first 10 values: {X_combined[0][:10]}")
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model feature count expected: {model.n_features_in_ if hasattr(model, 'n_features_in_') else 'unknown'}")
        
        # Check if the model feature count matches the input features
        if hasattr(model, 'n_features_in_') and X_combined.shape[1] != model.n_features_in_:
            logger.error(f"FEATURE COUNT MISMATCH: Model expects {model.n_features_in_} features but got {X_combined.shape[1]}")
        
        # Try to print model parameters
        try:
            logger.info(f"Model parameters: {model.get_params()}")
        except:
            logger.info("Could not get model parameters")
        
        # Make prediction
        try:
            prediction = model.predict_proba(X_combined)
            predicted_class = np.argmax(prediction, axis=1)[0]
            probabilities = prediction[0].tolist()
            
            # Log raw prediction data
            logger.info(f"Raw prediction probabilities: {probabilities}")
            logger.info(f"Sum of probabilities: {sum(probabilities)}")
            logger.info(f"Predicted class: {predicted_class}")
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}", exc_info=True)
            raise
        
        labels = {0: "Not popular", 1: "Controversy", 2: "Decent", 3: "Overwhelming positive"}
        
        if predicted_class == 3:
            recommendation = "absolutely recommend"
        elif predicted_class in [1, 2]:
            recommendation = "recommend"
        else:
            recommendation = "not recommend"
        
        response_data = {
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
        }
        
        # Log final response data
        logger.info(f"Response data: {json.dumps(response_data, indent=2)}")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error analyzing video: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
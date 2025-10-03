# Flask Backend for Tata Motors Customer Comment Analysis
# This application provides an API for analyzing customer comments using BERT (ML) and VADER (Rule-based) sentiment analysis

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from functools import lru_cache
import torch
import os
import logging
import re
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask application
app = Flask(__name__)
CORS(app)

# Global variables to hold the models
bert_classifier = None
vader_analyzer = None
dataset = None
intent_classifier = None
location_analytics = None

# Indian cities and states for location detection
INDIAN_CITIES = {
    'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Kolkata', 'Pune', 'Jaipur', 'Lucknow',
    'Kanpur', 'Nagpur', 'Indore', 'Thane', 'Bhopal', 'Visakhapatnam', 'Patna', 'Vadodara', 'Ghaziabad', 'Ludhiana',
    'Agra', 'Nashik', 'Faridabad', 'Meerut', 'Rajkot', 'Kalyan-Dombivali', 'Vasai-Virar', 'Varanasi', 'Srinagar',
    'Aurangabad', 'Dhanbad', 'Amritsar', 'Navi Mumbai', 'Allahabad', 'Ranchi', 'Howrah', 'Coimbatore', 'Jabalpur',
    'Gwalior', 'Vijayawada', 'Jodhpur', 'Madurai', 'Raipur', 'Kota', 'Guwahati', 'Chandigarh', 'Solapur', 'Hubballi-Dharwad',
    'Bareilly', 'Moradabad', 'Mysore', 'Gurgaon', 'Aligarh', 'Jalandhar', 'Tiruchirappalli', 'Bhubaneswar', 'Salem',
    'Warangal', 'Guntur', 'Bhiwandi', 'Saharanpur', 'Gorakhpur', 'Bikaner', 'Amravati', 'Noida', 'Jamshedpur',
    'Bhilai', 'Cuttack', 'Firozabad', 'Kochi', 'Bhavnagar', 'Dehradun', 'Durgapur', 'Asansol', 'Nanded-Waghala',
    'Kolhapur', 'Ajmer', 'Gulbarga', 'Jamnagar', 'Ujjain', 'Loni', 'Siliguri', 'Jhansi', 'Ulhasnagar', 'Nellore',
    'Jammu', 'Sangli-Miraj & Kupwad', 'Belgaum', 'Mangalore', 'Ambattur', 'Tirunelveli', 'Malegaon', 'Gaya',
    'Jalgaon', 'Udaipur', 'Maheshtala', 'Tirupur', 'Ongole', 'Bhagalpur', 'Muzaffarpur', 'Bhatpara', 'Panihati',
    'Latur', 'Dhule', 'Rohtak', 'Korba', 'Bhilwara', 'Berhampur', 'Muzaffarnagar', 'Ahmednagar', 'Mathura',
    'Kollam', 'Avadi', 'Kadapa', 'Kamarhati', 'Sambalpur', 'Bilaspur', 'Shahjahanpur', 'Satara', 'Bijapur',
    'Rampur', 'Shivamogga', 'Chandrapur', 'Junagadh', 'Thrissur', 'Alwar', 'Bardhaman', 'Kulti', 'Kakinada',
    'Nizamabad', 'Parbhani', 'Tumkur', 'Khammam', 'Ozhukarai', 'Bihar Sharif', 'Panipat', 'Darbhanga',
    'Bally', 'Aizawl', 'Dewas', 'Ichalkaranji', 'Karnal', 'Bathinda', 'Jalna', 'Eluru', 'Kirari Suleman Nagar',
    'Barasat', 'Purnia', 'Satna', 'Mau', 'Sonipat', 'Farrukhabad', 'Sagar', 'Rourkela', 'Durg', 'Imphal',
    'Ratlam', 'Hapur', 'Arrah', 'Anantapur', 'Karimnagar', 'Etawah', 'Ambernath', 'North Dumdum', 'Bharatpur',
    'Begusarai', 'New Delhi', 'Gandhidham', 'Baranagar', 'Tiruvottiyur', 'Puducherry', 'Sikar', 'Thoothukudi',
    'Rewa', 'Mirzapur', 'Raichur', 'Pali', 'Ramagundam', 'Silchar', 'Orai', 'Nandyal', 'Morena', 'Bhiwani',
    'Sambalpur', 'Bellary', 'Hospet', 'Karaikudi', 'Kishanganj', 'Puruliya', 'Kurnool', 'Rajpur Sonarpur'
}

INDIAN_STATES = {
    'Maharashtra', 'Tamil Nadu', 'Andhra Pradesh', 'Karnataka', 'Gujarat', 'West Bengal', 'Rajasthan',
    'Uttar Pradesh', 'Madhya Pradesh', 'Bihar', 'Odisha', 'Telangana', 'Kerala', 'Punjab', 'Haryana',
    'Jharkhand', 'Chhattisgarh', 'Assam', 'Delhi', 'Himachal Pradesh', 'Uttarakhand', 'Goa', 'Manipur',
    'Meghalaya', 'Tripura', 'Mizoram', 'Arunachal Pradesh', 'Nagaland', 'Sikkim', 'Jammu and Kashmir',
    'Ladakh', 'Chandigarh', 'Dadra and Nagar Haveli', 'Daman and Diu', 'Lakshadweep', 'Puducherry',
    'Andaman and Nicobar Islands'
}

# Regional mapping for analysis
REGIONAL_MAPPING = {
    'North': ['Delhi', 'Punjab', 'Haryana', 'Himachal Pradesh', 'Uttarakhand', 'Uttar Pradesh', 'Rajasthan', 'Jammu and Kashmir', 'Chandigarh'],
    'South': ['Tamil Nadu', 'Karnataka', 'Andhra Pradesh', 'Telangana', 'Kerala', 'Puducherry', 'Lakshadweep'],
    'West': ['Maharashtra', 'Gujarat', 'Goa', 'Dadra and Nagar Haveli', 'Daman and Diu'],
    'East': ['West Bengal', 'Odisha', 'Jharkhand', 'Bihar', 'Andaman and Nicobar Islands'],
    'Central': ['Madhya Pradesh', 'Chhattisgarh'],
    'Northeast': ['Assam', 'Meghalaya', 'Manipur', 'Tripura', 'Mizoram', 'Arunachal Pradesh', 'Nagaland', 'Sikkim']
}


def load_dataset():
    """
    Load the Tata Motors customer comments dataset.
    Returns the loaded DataFrame.
    """
    try:
        df = pd.read_csv("synthetic_tata_motors_data.csv")
        logger.info(f"Dataset loaded successfully with {len(df)} records")

        # Handle missing values
        df = df.dropna(subset=["text", "category"])

        return df
    except FileNotFoundError:
        logger.error("Dataset file not found. Creating mock dataset.")
        # Create mock data if CSV doesn't exist
        mock_data = {
            "id": [1, 2, 3, 4, 5, 6],
            "text": [
                "The service at the dealership was terrible and unprofessional",
                "I absolutely love my Nexon EV! Amazing features and performance",
                "The build quality could be improved, especially the interior plastics",
                "Great value for money, would definitely recommend to others",
                "The charging infrastructure needs serious improvement across the country",
                "The staff was very helpful and professional during my visit",
            ],
            "category": [
                "Complaint / Criticism",
                "Praise / Satisfaction",
                "Suggestion / Feature Request",
                "Praise / Satisfaction",
                "Suggestion / Feature Request",
                "Praise / Satisfaction",
            ],
        }
        return pd.DataFrame(mock_data)


def initialize_bert_model():
    """
    Initialize BERT model for sentiment analysis.
    Returns the initialized pipeline.
    """
    try:
        # Use a pre-trained sentiment analysis model
        model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

        # Initialize the pipeline
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            tokenizer=model_name,
            device=0 if torch.cuda.is_available() else -1,
        )

        logger.info("BERT model initialized successfully")
        return sentiment_pipeline

    except Exception as e:
        logger.error(f"Error initializing BERT model: {str(e)}")
        # Fallback to a simpler model
        try:
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1,
            )
            logger.info("Fallback BERT model initialized")
            return sentiment_pipeline
        except Exception as fallback_error:
            logger.error(f"Fallback model also failed: {str(fallback_error)}")
            return None


def initialize_vader():
    """
    Initialize VADER sentiment analyzer.
    """
    try:
        analyzer = SentimentIntensityAnalyzer()
        logger.info("VADER analyzer initialized successfully")
        return analyzer
    except Exception as e:
        logger.error(f"Error initializing VADER: {str(e)}")
        return None


def train_intent_classifier(df):
    """
    Train a model to classify the comment's intent/category.
    """
    logger.info("Training intent classification model...")

    try:
        # Validate required columns exist
        if "text" not in df.columns or "category" not in df.columns:
            logger.error("Dataset missing required columns: 'text' and/or 'category'")
            return None

        # Remove any rows with missing values
        df_clean = df.dropna(subset=["text", "category"])

        if len(df_clean) == 0:
            logger.error("No valid training data available after cleaning")
            return None

        logger.info(
            f"Training on {len(df_clean)} samples with {len(df_clean['category'].unique())} categories"
        )

        # Define features (X) and target (y)
        X = df_clean["text"]
        y = df_clean["category"]

        # Create a model pipeline
        intent_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
                ("clf", LogisticRegression(solver="liblinear", random_state=42)),
            ]
        )

        # Train the model
        intent_pipeline.fit(X, y)

        logger.info("Intent classification model trained successfully.")
        logger.info(f"Model can predict categories: {list(intent_pipeline.classes_)}")
        return intent_pipeline

    except Exception as e:
        logger.error(f"Error training intent classifier: {str(e)}")
        return None


@lru_cache(maxsize=128)
def detect_location(text):
    """
    Detect locations mentioned in customer comments.
    
    Args:
        text (str): The customer comment to analyze
        
    Returns:
        dict: Detected locations with cities, states, and regions
    """
    text_upper = text.upper()
    detected_cities = []
    detected_states = []
    detected_regions = set()
    
    # Detect cities
    for city in INDIAN_CITIES:
        if city.upper() in text_upper:
            detected_cities.append(city)
            # Map city to region
            for region, states in REGIONAL_MAPPING.items():
                # Find which state this city belongs to (simplified mapping)
                if city in ['Mumbai', 'Pune', 'Nagpur', 'Nashik', 'Aurangabad']:
                    if 'Maharashtra' in states:
                        detected_regions.add(region)
                elif city in ['Delhi', 'New Delhi', 'Gurgaon', 'Faridabad', 'Noida']:
                    if 'Delhi' in states or 'Haryana' in states or 'Uttar Pradesh' in states:
                        detected_regions.add(region)
                elif city in ['Bangalore', 'Mysore']:
                    if 'Karnataka' in states:
                        detected_regions.add(region)
                elif city in ['Chennai', 'Madurai', 'Coimbatore']:
                    if 'Tamil Nadu' in states:
                        detected_regions.add(region)
                elif city in ['Kolkata', 'Howrah']:
                    if 'West Bengal' in states:
                        detected_regions.add(region)
                elif city in ['Hyderabad']:
                    if 'Telangana' in states:
                        detected_regions.add(region)
                elif city in ['Ahmedabad', 'Rajkot', 'Vadodara']:
                    if 'Gujarat' in states:
                        detected_regions.add(region)
                elif city in ['Jaipur', 'Jodhpur', 'Udaipur']:
                    if 'Rajasthan' in states:
                        detected_regions.add(region)
    
    # Detect states
    for state in INDIAN_STATES:
        if state.upper() in text_upper:
            detected_states.append(state)
            # Map state to region
            for region, states in REGIONAL_MAPPING.items():
                if state in states:
                    detected_regions.add(region)
    
    return {
        'cities': detected_cities,
        'states': detected_states,
        'regions': list(detected_regions),
        'has_location': bool(detected_cities or detected_states)
    }

@lru_cache(maxsize=128)
def analyze_aspects(text):
    """
    Analyze the business aspects mentioned in a customer comment.

    Args:
        text (str): The customer comment to analyze

    Returns:
        list: List of identified business aspects
    """
    aspect_keywords = {
        "Service": [
            "service",
            "dealer",
            "dealership",
            "staff",
            "repair",
            "maintenance",
            "support",
            "customer care",
            "after-sales",
            "service center",
            "technician",
        ],
        "Features": [
            "features",
            "infotainment",
            "technology",
            "safety",
            "airbag",
            "abs",
            "touchscreen",
            "connectivity",
            "bluetooth",
            "android auto",
            "carplay",
        ],
        "EV": [
            "ev",
            "nexon ev",
            "tiago ev",
            "charging",
            "range",
            "battery",
            "electric",
            "charge",
            "mileage per charge",
            "charging station",
            "battery life",
        ],
        "Price": [
            "price",
            "cost",
            "expensive",
            "cheap",
            "value",
            "money",
            "budget",
            "affordable",
            "pricing",
            "value for money",
            "overpriced",
            "worth",
        ],
        "Build Quality": [
            "build",
            "quality",
            "construction",
            "material",
            "durability",
            "solid",
            "sturdy",
            "plastic",
            "fit",
            "finish",
            "panel gaps",
            "interior quality",
        ],
        "Performance": [
            "performance",
            "speed",
            "acceleration",
            "power",
            "engine",
            "smooth",
            "handling",
            "drive",
            "pickup",
            "torque",
            "responsive",
        ],
        "Design": [
            "design",
            "look",
            "appearance",
            "styling",
            "beautiful",
            "ugly",
            "attractive",
            "interior",
            "exterior",
            "dashboard",
            "seats",
        ],
        "Fuel Efficiency": [
            "mileage",
            "fuel",
            "efficiency",
            "consumption",
            "economy",
            "petrol",
            "diesel",
            "fuel economy",
            "kmpl",
        ],
        "Sound System": [
            "sound",
            "audio",
            "music",
            "harman",
            "speakers",
            "bass",
            "treble",
            "sound system",
            "acoustics",
        ],
        "Comfort": [
            "comfort",
            "comfortable",
            "seat",
            "seating",
            "space",
            "legroom",
            "headroom",
            "ergonomic",
            "cushioning",
            "ventilated seats",
        ],
    }

    text_lower = text.lower()
    identified_aspects = []

    for aspect, keywords in aspect_keywords.items():
        if any(keyword in text_lower for keyword in keywords):
            identified_aspects.append(aspect)

    return identified_aspects


@lru_cache(maxsize=128)
def bert_sentiment_analysis(text):
    """
    Perform sentiment analysis using BERT model.

    Args:
        text (str): Text to analyze

    Returns:
        dict: BERT sentiment analysis results
    """
    if bert_classifier is None:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "error": "BERT model not available",
        }

    try:
        result = bert_classifier(text)[0]

        # Map the labels to standardized format
        label_mapping = {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
            "NEUTRAL": "neutral",
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        }

        sentiment = label_mapping.get(result["label"], result["label"].lower())
        confidence = result["score"]

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "raw_output": result,
        }

    except Exception as e:
        logger.error(f"BERT analysis error: {str(e)}")
        return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}


def vader_sentiment_analysis(text):
    """
    Perform sentiment analysis using VADER.

    Args:
        text (str): Text to analyze

    Returns:
        dict: VADER sentiment analysis results
    """
    if vader_analyzer is None:
        return {
            "sentiment": "neutral",
            "scores": {},
            "error": "VADER analyzer not available",
        }

    try:
        scores = vader_analyzer.polarity_scores(text)

        # Determine overall sentiment
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"

        return {
            "sentiment": sentiment,
            "scores": {
                "positive": round(scores["pos"], 4),
                "neutral": round(scores["neu"], 4),
                "negative": round(scores["neg"], 4),
                "compound": round(scores["compound"], 4),
            },
        }

    except Exception as e:
        logger.error(f"VADER analysis error: {str(e)}")
        return {"sentiment": "neutral", "scores": {}, "error": str(e)}


# Recommendation rules based on category and aspect combinations
RECOMMENDATION_RULES = {
    ("Complaint / Criticism", "Service"): {
        "insight": "Customers are reporting negative experiences with dealership service...",
        "strategy": "Service Excellence Initiative",
        "action": "Immediately review service center protocols and staff training programs...",
        "priority": "High",
    },
    ("Praise / Satisfaction", "Features"): {
        "insight": "Customers highly appreciate current feature offerings...",
        "strategy": "Feature Leadership Amplification",
        "action": "Leverage positive feedback in marketing campaigns and product showcases...",
        "priority": "Medium",
    },
    ("Complaint / Criticism", "Price"): {
        "insight": "Customers expressing concerns about pricing and value proposition...",
        "strategy": "Value Communication Strategy",
        "action": "Review pricing structure and enhance value messaging in communications...",
        "priority": "High",
    },
    ("Praise / Satisfaction", "Performance"): {
        "insight": "Customers are satisfied with vehicle performance characteristics...",
        "strategy": "Performance Excellence Showcase",
        "action": "Highlight performance strengths in product communications and testimonials...",
        "priority": "Medium",
    },
    ("Inquiry / Question", "Features"): {
        "insight": "Customers seeking more information about available features...",
        "strategy": "Feature Education Campaign",
        "action": "Develop comprehensive feature guides and interactive demonstrations...",
        "priority": "Low",
    },
    ("Complaint / Criticism", "Quality"): {
        "insight": "Customers reporting quality issues with vehicles...",
        "strategy": "Quality Assurance Enhancement",
        "action": "Investigate quality control processes and implement corrective measures...",
        "priority": "High",
    },
    ("Praise / Satisfaction", "Design"): {
        "insight": "Customers are praising vehicle design and aesthetics...",
        "strategy": "Design Excellence Promotion",
        "action": "Feature design highlights in marketing materials and showrooms...",
        "priority": "Medium",
    },
    ("Suggestion / Feature Request", "Technology"): {
        "insight": "Customers requesting advanced technology features...",
        "strategy": "Innovation Roadmap Development",
        "action": "Evaluate feasibility of requested tech features for future models...",
        "priority": "Medium",
    },
    ("Complaint / Criticism", "Comfort"): {
        "insight": "Customers reporting comfort-related issues...",
        "strategy": "Comfort Enhancement Initiative",
        "action": "Review seat design and cabin ergonomics for improvements...",
        "priority": "Medium",
    },
    ("Praise / Satisfaction", "Safety"): {
        "insight": "Customers appreciating safety features and ratings...",
        "strategy": "Safety Leadership Messaging",
        "action": "Promote safety achievements and certifications in marketing...",
        "priority": "Medium",
    },
    ("Competitive Comparison", "Features"): {
        "insight": "Customers comparing features with competitor vehicles...",
        "strategy": "Competitive Differentiation",
        "action": "Analyze competitor features and strengthen unique value propositions...",
        "priority": "Medium",
    },
    ("Inquiry / Question", "Price"): {
        "insight": "Customers inquiring about pricing and financing options...",
        "strategy": "Pricing Transparency Initiative",
        "action": "Improve pricing communication and financing option visibility...",
        "priority": "Low",
    },
    ("Suggestion / Feature Request", "Design"): {
        "insight": "Customers suggesting design improvements or modifications...",
        "strategy": "Customer-Centric Design Evolution",
        "action": "Document design suggestions for consideration in future model updates...",
        "priority": "Low",
    },
    ("Complaint / Criticism", "Technology"): {
        "insight": "Customers reporting issues with technology features...",
        "strategy": "Technology Enhancement Program",
        "action": "Review and upgrade software/hardware systems based on feedback...",
        "priority": "High",
    },
    ("Praise / Satisfaction", "Service"): {
        "insight": "Customers praising excellent service experiences...",
        "strategy": "Service Excellence Recognition",
        "action": "Recognize outstanding service teams and replicate best practices...",
        "priority": "Low",
    },
}

# Default recommendation for cases not covered by specific rules
DEFAULT_RECOMMENDATION = {
    "insight": "Customer feedback requires individual assessment...",
    "strategy": "Personalized Response Strategy",
    "action": "Route to appropriate customer service team for detailed review...",
    "priority": "Medium",
}


def generate_location_insights(dataset_with_location):
    """
    Generate comprehensive location-based insights from the dataset.
    
    Args:
        dataset_with_location (pd.DataFrame): Dataset with location column
        
    Returns:
        dict: Comprehensive location analytics
    """
    if dataset_with_location.empty:
        return {"error": "No location data available"}
    
    insights = {
        "city_analytics": {},
        "regional_analytics": {},
        "location_sentiment": {},
        "geographic_trends": {},
        "dealership_performance": {}
    }
    
    # City-wise analytics
    city_data = dataset_with_location.groupby('location').agg({
        'text': 'count',
        'category': lambda x: x.value_counts().to_dict()
    }).to_dict()
    
    for city in dataset_with_location['location'].unique():
        city_comments = dataset_with_location[dataset_with_location['location'] == city]
        
        # Sentiment analysis for each city
        city_sentiments = []
        for comment in city_comments['text']:
            bert_result = bert_sentiment_analysis(comment)
            city_sentiments.append(bert_result['sentiment'])
        
        sentiment_counts = Counter(city_sentiments)
        total_comments = len(city_comments)
        
        insights["city_analytics"][city] = {
            "total_comments": total_comments,
            "sentiment_distribution": dict(sentiment_counts),
            "sentiment_percentages": {
                sentiment: round((count / total_comments) * 100, 1)
                for sentiment, count in sentiment_counts.items()
            },
            "category_distribution": city_comments['category'].value_counts().to_dict(),
            "dominant_sentiment": sentiment_counts.most_common(1)[0][0] if sentiment_counts else "neutral",
            "engagement_score": calculate_engagement_score(city_comments)
        }
    
    # Regional analytics
    for region, states in REGIONAL_MAPPING.items():
        regional_cities = []
        for city in insights["city_analytics"].keys():
            # Simplified region mapping based on major cities
            if city in ['Mumbai', 'Pune', 'Nagpur']:
                if region == 'West':
                    regional_cities.append(city)
            elif city in ['Delhi', 'Jaipur', 'Chandigarh']:
                if region == 'North':
                    regional_cities.append(city)
            elif city in ['Bangalore', 'Chennai', 'Hyderabad']:
                if region == 'South':
                    regional_cities.append(city)
            elif city in ['Kolkata', 'Bhubaneswar']:
                if region == 'East':
                    regional_cities.append(city)
            elif city in ['Indore', 'Bhopal']:
                if region == 'Central':
                    regional_cities.append(city)
            elif city in ['Guwahati']:
                if region == 'Northeast':
                    regional_cities.append(city)
        
        if regional_cities:
            regional_data = {
                "cities": regional_cities,
                "total_comments": sum(insights["city_analytics"][city]["total_comments"] for city in regional_cities),
                "avg_engagement": sum(insights["city_analytics"][city]["engagement_score"] for city in regional_cities) / len(regional_cities),
                "dominant_categories": Counter()
            }
            
            # Aggregate sentiment across region
            regional_sentiments = Counter()
            for city in regional_cities:
                city_sentiments = insights["city_analytics"][city]["sentiment_distribution"]
                for sentiment, count in city_sentiments.items():
                    regional_sentiments[sentiment] += count
            
            regional_data["sentiment_distribution"] = dict(regional_sentiments)
            insights["regional_analytics"][region] = regional_data
    
    # Geographic trends (identify patterns)
    insights["geographic_trends"] = {
        "high_satisfaction_cities": [
            city for city, data in insights["city_analytics"].items()
            if data["sentiment_percentages"].get("positive", 0) > 60
        ],
        "improvement_needed_cities": [
            city for city, data in insights["city_analytics"].items()
            if data["sentiment_percentages"].get("negative", 0) > 40
        ],
        "service_hotspots": identify_service_hotspots(dataset_with_location),
        "growth_markets": identify_growth_markets(insights["city_analytics"])
    }
    
    return insights

def calculate_engagement_score(city_comments):
    """Calculate engagement score based on comment characteristics."""
    if len(city_comments) == 0:
        return 0
    
    # Simple engagement score based on comment length and variety
    avg_length = city_comments['text'].str.len().mean()
    category_variety = len(city_comments['category'].unique())
    
    # Normalize and combine metrics
    length_score = min(avg_length / 100, 1.0)  # Normalize to 0-1
    variety_score = min(category_variety / 5, 1.0)  # Max 5 categories
    
    return round((length_score * 0.6 + variety_score * 0.4) * 100, 1)

def identify_service_hotspots(dataset_with_location):
    """Identify cities with high service-related feedback."""
    service_keywords = ['service', 'dealer', 'dealership', 'staff', 'repair', 'maintenance']
    service_hotspots = []
    
    for city in dataset_with_location['location'].unique():
        city_comments = dataset_with_location[dataset_with_location['location'] == city]
        service_mentions = 0
        
        for comment in city_comments['text']:
            if any(keyword in comment.lower() for keyword in service_keywords):
                service_mentions += 1
        
        if service_mentions > 0:
            service_ratio = service_mentions / len(city_comments)
            if service_ratio > 0.3:  # More than 30% service mentions
                service_hotspots.append({
                    "city": city,
                    "service_mention_ratio": round(service_ratio * 100, 1),
                    "total_comments": len(city_comments)
                })
    
    return sorted(service_hotspots, key=lambda x: x["service_mention_ratio"], reverse=True)

def identify_growth_markets(city_analytics):
    """Identify potential growth markets based on sentiment and engagement."""
    growth_markets = []
    
    for city, data in city_analytics.items():
        positive_ratio = data["sentiment_percentages"].get("positive", 0)
        engagement = data["engagement_score"]
        comment_volume = data["total_comments"]
        
        # Growth potential based on positive sentiment but moderate volume
        if positive_ratio > 50 and engagement > 60 and comment_volume < 20:
            growth_markets.append({
                "city": city,
                "positive_sentiment": positive_ratio,
                "engagement_score": engagement,
                "comment_volume": comment_volume,
                "growth_potential": "High"
            })
    
    return sorted(growth_markets, key=lambda x: x["positive_sentiment"], reverse=True)

def generate_location_recommendation(category, aspects, location_data, bert_sentiment, vader_sentiment):
    """
    Generate location-aware strategic recommendations.
    
    Args:
        category (str): The predicted category of the comment
        aspects (list): List of identified business aspects
        location_data (dict): Detected location information
        bert_sentiment (dict): BERT sentiment analysis results
        vader_sentiment (dict): VADER sentiment analysis results
    
    Returns:
        dict: Location-aware strategic recommendation
    """
    base_recommendation = generate_recommendation(category, aspects, bert_sentiment, vader_sentiment)
    
    if not location_data['has_location']:
        return base_recommendation
    
    # Enhance recommendation with location-specific insights
    cities = location_data.get('cities', [])
    regions = location_data.get('regions', [])
    
    location_context = ""
    location_actions = []
    
    if cities:
        primary_city = cities[0]
        location_context = f"Feedback from {primary_city} market indicates "
        
        # City-specific recommendations
        metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
        tier2_cities = ['Pune', 'Jaipur', 'Ahmedabad', 'Lucknow', 'Kanpur', 'Nagpur']
        
        if primary_city in metro_cities:
            location_actions.append(f"Prioritize {primary_city} metro market initiatives")
            location_actions.append("Leverage urban dealership network for immediate response")
        elif primary_city in tier2_cities:
            location_actions.append(f"Strengthen {primary_city} tier-2 market presence")
            location_actions.append("Focus on regional dealership training and support")
        else:
            location_actions.append(f"Develop {primary_city} emerging market strategy")
            location_actions.append("Consider service network expansion in this region")
    
    if regions:
        primary_region = regions[0]
        location_actions.append(f"Implement region-wide strategy for {primary_region} markets")
        
        # Region-specific strategies
        if primary_region == "South":
            location_actions.append("Leverage strong South Indian market base for expansion")
        elif primary_region == "West":
            location_actions.append("Capitalize on West region's commercial vehicle demand")
        elif primary_region == "North":
            location_actions.append("Address North region's diverse customer preferences")
    
    # Enhance base recommendation with location insights
    enhanced_recommendation = base_recommendation.copy()
    enhanced_recommendation["insight"] = location_context + base_recommendation["insight"].lower()
    enhanced_recommendation["action"] = base_recommendation["action"] + " " + " ".join(location_actions)
    enhanced_recommendation["location_context"] = {
        "detected_locations": location_data,
        "geographic_focus": cities[0] if cities else regions[0] if regions else "National",
        "market_type": "Metro" if cities and cities[0] in ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad'] else "Regional"
    }
    
    return enhanced_recommendation


def generate_recommendation_v2(
    category, aspects, bert_sentiment=None, vader_sentiment=None
):
    """
    Generate strategic recommendations using rule-based approach.

    Args:
        category (str): The predicted category of the comment
        aspects (list): List of identified business aspects
        bert_sentiment (dict, optional): BERT sentiment analysis results
        vader_sentiment (dict, optional): VADER sentiment analysis results

    Returns:
        dict: Strategic recommendation with insight, strategy, action, and priority
    """
    for aspect in aspects:
        if (category, aspect) in RECOMMENDATION_RULES:
            # Return the first rule that matches
            return RECOMMENDATION_RULES[(category, aspect)]

    # Return a default recommendation if no specific rule is found
    return DEFAULT_RECOMMENDATION


def generate_recommendation(category, aspects, bert_sentiment, vader_sentiment):
    """
    Generate strategic recommendations based on category, aspects, and sentiment analysis.

    Args:
        category (str): The predicted category of the comment
        aspects (list): List of identified business aspects
        bert_sentiment (dict): BERT sentiment analysis results
        vader_sentiment (dict): VADER sentiment analysis results

    Returns:
        dict: Strategic recommendation with insight, strategy, action, and priority
    """
    # Determine overall sentiment consensus
    bert_sent = bert_sentiment.get("sentiment", "neutral")
    vader_sent = vader_sentiment.get("sentiment", "neutral")

    # Priority mapping based on sentiment and category
    if category in ["Complaint / Criticism"] or (
        bert_sent == "negative" and vader_sent == "negative"
    ):
        base_priority = "High"
    elif category in ["Suggestion / Feature Request"]:
        base_priority = "Medium"
    else:
        base_priority = "Low"

    # Generate recommendations based on category and aspects
    if category == "Complaint / Criticism" or bert_sent == "negative":
        if "Service" in aspects:
            return {
                "insight": "Customers are reporting negative experiences with dealership service and after-sales support.",
                "strategy": "Service Excellence Initiative",
                "action": "Immediately review service center protocols, conduct staff retraining, and implement customer feedback loop.",
                "priority": "High",
                "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
            }
        elif "EV" in aspects:
            return {
                "insight": "Customers facing challenges with EV-related features, charging, or range anxiety.",
                "strategy": "EV Experience Enhancement",
                "action": "Investigate EV-specific issues, improve charging infrastructure partnerships, and enhance customer education.",
                "priority": "High",
                "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
            }
        elif "Build Quality" in aspects:
            return {
                "insight": "Customers expressing dissatisfaction with build quality and materials used.",
                "strategy": "Quality Assurance Overhaul",
                "action": "Escalate to manufacturing team for comprehensive quality audit and supplier assessment.",
                "priority": "High",
                "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
            }

    elif category == "Praise / Satisfaction" or bert_sent == "positive":
        if "Features" in aspects:
            return {
                "insight": "Customers highly appreciate current feature offerings and technological innovations.",
                "strategy": "Feature Leadership Amplification",
                "action": "Leverage positive feedback in marketing campaigns and accelerate similar feature development.",
                "priority": "Medium",
                "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
            }
        elif "Service" in aspects:
            return {
                "insight": "Customers satisfied with service quality at specific touchpoints.",
                "strategy": "Service Excellence Replication",
                "action": "Document and replicate successful service practices across all dealership networks.",
                "priority": "Medium",
                "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
            }

    elif category == "Suggestion / Feature Request":
        if "Features" in aspects:
            return {
                "insight": "Customers actively suggesting new features and improvements.",
                "strategy": "Customer-Driven Innovation",
                "action": "Prioritize suggested features in product roadmap and establish customer co-creation program.",
                "priority": "Medium",
                "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
            }

    elif category == "Purchase Intent / Inquiry":
        return {
            "insight": "Customer showing interest in purchase and seeking detailed information.",
            "strategy": "Sales Conversion Optimization",
            "action": "Ensure sales team follows up promptly with comprehensive product information and test drive offers.",
            "priority": "High",
            "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
        }

    elif category == "Competitive Comparison":
        return {
            "insight": "Customer comparing Tata Motors products with competitors.",
            "strategy": "Competitive Positioning",
            "action": "Analyze competitor advantages mentioned and strengthen unique value propositions in marketing.",
            "priority": "Medium",
            "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
        }

    # Default recommendation
    return {
        "insight": "General customer feedback received requiring attention and follow-up.",
        "strategy": "Customer Relationship Management",
        "action": "Follow up with customer to gather more specific feedback and ensure satisfaction.",
        "priority": base_priority,
        "sentiment_consensus": f"BERT: {bert_sent}, VADER: {vader_sent}",
    }


@app.route("/analyze", methods=["POST"])
def analyze_comment():
    """
    API endpoint to analyze customer comments using BERT and VADER sentiment analysis.

    Expected JSON payload: {"text": "customer comment text"}

    Returns:
        JSON response with comprehensive analysis results including location data
    """
    try:
        # Get JSON payload
        data = request.get_json()

        # Validate input
        if not data or "text" not in data:
            return jsonify({"error": "Missing required field: text"}), 400

        comment_text = data["text"].strip()

        if not comment_text:
            return jsonify({"error": "Text cannot be empty"}), 400

        # Perform BERT sentiment analysis
        bert_analysis = bert_sentiment_analysis(comment_text)

        # Perform VADER sentiment analysis
        vader_analysis = vader_sentiment_analysis(comment_text)

        # Analyze business aspects
        identified_aspects = analyze_aspects(comment_text)

        # Detect locations mentioned in the comment
        location_data = detect_location(comment_text)

        # Use the trained model for prediction
        if intent_classifier is not None:
            predicted_category = intent_classifier.predict([comment_text])[0]
        else:
            # Fallback to keyword matching if model not available
            predicted_category = predict_category_fallback(comment_text)

        # Generate location-aware strategic recommendation
        if location_data['has_location']:
            strategic_recommendation = generate_location_recommendation(
                predicted_category, identified_aspects, location_data, bert_analysis, vader_analysis
            )
        else:
            # Use original recommendation system if no location detected
            strategic_recommendation = generate_recommendation_v2(
                predicted_category, identified_aspects, bert_analysis, vader_analysis
            )
            
            # Fallback to original recommendation if new approach returns default
            if strategic_recommendation == DEFAULT_RECOMMENDATION:
                strategic_recommendation = generate_recommendation(
                    predicted_category, identified_aspects, bert_analysis, vader_analysis
                )

        # Structure the response
        response = {
            "input_text": comment_text,
            "predicted_category": predicted_category,
            "identified_aspects": identified_aspects,
            "location_analysis": location_data,
            "sentiment_analysis": {"bert": bert_analysis, "vader": vader_analysis},
            "strategic_recommendation": strategic_recommendation,
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify(
            {"error": f"An error occurred while processing the request: {str(e)}"}
        ), 500


def predict_category_fallback(text):
    """
    Simple keyword-based category prediction as fallback.
    """
    text_lower = text.lower()

    # Complaint indicators
    complaint_words = [
        "terrible",
        "bad",
        "worst",
        "hate",
        "awful",
        "disappointed",
        "problem",
        "issue",
        "complain",
    ]
    if any(word in text_lower for word in complaint_words):
        return "Complaint / Criticism"

    # Praise indicators
    praise_words = [
        "love",
        "great",
        "excellent",
        "amazing",
        "fantastic",
        "best",
        "good",
        "awesome",
        "recommend",
    ]
    if any(word in text_lower for word in praise_words):
        return "Praise / Satisfaction"

    # Suggestion indicators
    suggestion_words = [
        "should",
        "could",
        "wish",
        "hope",
        "suggest",
        "improve",
        "better",
        "add",
        "include",
    ]
    if any(word in text_lower for word in suggestion_words):
        return "Suggestion / Feature Request"

    # Purchase intent indicators
    purchase_words = [
        "buy",
        "purchase",
        "confused",
        "which",
        "variant",
        "price",
        "cost",
        "thinking",
    ]
    if any(word in text_lower for word in purchase_words):
        return "Purchase Intent / Inquiry"

    # Comparison indicators
    comparison_words = [
        "vs",
        "versus",
        "compared",
        "better than",
        "hyundai",
        "maruti",
        "kia",
    ]
    if any(word in text_lower for word in comparison_words):
        return "Competitive Comparison"

    return "General Feedback"


@app.route("/location-analytics", methods=["GET"])
def get_location_analytics():
    """
    API endpoint to get comprehensive location-based analytics.
    
    Returns:
        JSON response with location analytics from the dataset
    """
    try:
        if dataset is None:
            return jsonify({"error": "Dataset not loaded"}), 500
        
        # Generate location insights from the dataset
        location_insights = generate_location_insights(dataset)
        
        return jsonify({
            "status": "success",
            "data": location_insights,
            "total_locations": len(location_insights.get("city_analytics", {})),
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Location analytics error: {str(e)}")
        return jsonify({"error": f"Failed to generate location analytics: {str(e)}"}), 500

@app.route("/city-trends", methods=["GET"])
def get_city_trends():
    """
    API endpoint to get city-wise sentiment trends and patterns.
    
    Query parameters:
        - city: Filter by specific city (optional)
        - limit: Number of top cities to return (default: 10)
    
    Returns:
        JSON response with city trends data
    """
    try:
        if dataset is None:
            return jsonify({"error": "Dataset not loaded"}), 500
        
        city_filter = request.args.get('city')
        limit = int(request.args.get('limit', 10))
        
        # Get location insights
        location_insights = generate_location_insights(dataset)
        city_analytics = location_insights.get("city_analytics", {})
        
        if city_filter:
            # Filter for specific city
            if city_filter in city_analytics:
                city_data = {city_filter: city_analytics[city_filter]}
            else:
                return jsonify({"error": f"City '{city_filter}' not found in dataset"}), 404
        else:
            # Get top cities by comment volume
            sorted_cities = sorted(
                city_analytics.items(),
                key=lambda x: x[1]["total_comments"],
                reverse=True
            )[:limit]
            city_data = dict(sorted_cities)
        
        # Calculate trends
        trends = {
            "city_performance": city_data,
            "top_positive_cities": [
                {"city": city, "positive_percentage": data["sentiment_percentages"].get("positive", 0)}
                for city, data in sorted(
                    city_analytics.items(),
                    key=lambda x: x[1]["sentiment_percentages"].get("positive", 0),
                    reverse=True
                )[:5]
            ],
            "improvement_opportunities": [
                {"city": city, "negative_percentage": data["sentiment_percentages"].get("negative", 0)}
                for city, data in sorted(
                    city_analytics.items(),
                    key=lambda x: x[1]["sentiment_percentages"].get("negative", 0),
                    reverse=True
                )[:5]
            ],
            "engagement_leaders": [
                {"city": city, "engagement_score": data["engagement_score"]}
                for city, data in sorted(
                    city_analytics.items(),
                    key=lambda x: x[1]["engagement_score"],
                    reverse=True
                )[:5]
            ]
        }
        
        return jsonify({
            "status": "success",
            "data": trends,
            "filter_applied": {"city": city_filter, "limit": limit},
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"City trends error: {str(e)}")
        return jsonify({"error": f"Failed to generate city trends: {str(e)}"}), 500

@app.route("/regional-performance", methods=["GET"])
def get_regional_performance():
    """
    API endpoint to get regional performance analytics.
    
    Returns:
        JSON response with regional performance data
    """
    try:
        if dataset is None:
            return jsonify({"error": "Dataset not loaded"}), 500
        
        # Get location insights
        location_insights = generate_location_insights(dataset)
        regional_analytics = location_insights.get("regional_analytics", {})
        
        # Calculate regional rankings
        regional_performance = {}
        for region, data in regional_analytics.items():
            if data["total_comments"] > 0:
                sentiment_dist = data["sentiment_distribution"]
                total = sum(sentiment_dist.values()) if sentiment_dist else 1
                
                regional_performance[region] = {
                    "total_comments": data["total_comments"],
                    "cities_covered": len(data["cities"]),
                    "cities": data["cities"],
                    "avg_engagement": data["avg_engagement"],
                    "sentiment_percentages": {
                        sentiment: round((count / total) * 100, 1)
                        for sentiment, count in sentiment_dist.items()
                    },
                    "performance_score": calculate_regional_performance_score(data, sentiment_dist, total)
                }
        
        # Rank regions by performance
        ranked_regions = sorted(
            regional_performance.items(),
            key=lambda x: x[1]["performance_score"],
            reverse=True
        )
        
        return jsonify({
            "status": "success",
            "data": {
                "regional_performance": regional_performance,
                "regional_rankings": [{"region": region, "score": data["performance_score"]} for region, data in ranked_regions],
                "best_performing_region": ranked_regions[0][0] if ranked_regions else None,
                "improvement_needed_region": ranked_regions[-1][0] if ranked_regions else None
            },
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Regional performance error: {str(e)}")
        return jsonify({"error": f"Failed to generate regional performance: {str(e)}"}), 500

def calculate_regional_performance_score(data, sentiment_dist, total):
    """Calculate a performance score for a region based on multiple factors."""
    if total == 0:
        return 0
    
    # Factors: positive sentiment %, engagement score, comment volume
    positive_ratio = sentiment_dist.get('positive', 0) / total
    engagement_normalized = data["avg_engagement"] / 100  # Normalize to 0-1
    volume_normalized = min(data["total_comments"] / 50, 1.0)  # Normalize, cap at 50 comments
    
    # Weighted score
    score = (positive_ratio * 0.5) + (engagement_normalized * 0.3) + (volume_normalized * 0.2)
    return round(score * 100, 1)

@app.route("/location-insights", methods=["POST"])
def get_location_insights():
    """
    API endpoint to get insights for a specific location mentioned in text.
    
    Expected JSON payload: {"text": "comment with location", "location": "optional specific location"}
    
    Returns:
        JSON response with location-specific insights
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        text = data.get("text", "")
        specific_location = data.get("location")
        
        if not text and not specific_location:
            return jsonify({"error": "Either text or location must be provided"}), 400
        
        # Detect location from text if not provided
        if text:
            location_data = detect_location(text)
        else:
            location_data = {"cities": [specific_location] if specific_location else [], "has_location": bool(specific_location)}
        
        if not location_data["has_location"]:
            return jsonify({"message": "No location detected in the provided text"})
        
        # Get dataset insights for detected locations
        dataset_insights = generate_location_insights(dataset)
        
        # Extract relevant insights for detected locations
        relevant_insights = {}
        for city in location_data.get("cities", []):
            if city in dataset_insights.get("city_analytics", {}):
                relevant_insights[city] = dataset_insights["city_analytics"][city]
        
        # Generate location-specific recommendations
        recommendations = []
        for city in location_data.get("cities", []):
            if city in relevant_insights:
                city_data = relevant_insights[city]
                dominant_sentiment = city_data["dominant_sentiment"]
                
                if dominant_sentiment == "negative":
                    recommendations.append(f"Immediate attention needed in {city} - high negative sentiment detected")
                elif dominant_sentiment == "positive":
                    recommendations.append(f"Leverage {city} as a success model for other markets")
                else:
                    recommendations.append(f"Monitor {city} market for emerging trends")
        
        return jsonify({
            "status": "success",
            "data": {
                "detected_locations": location_data,
                "city_insights": relevant_insights,
                "recommendations": recommendations,
                "geographic_context": {
                    "market_type": determine_market_type(location_data.get("cities", [])),
                    "regional_priority": determine_regional_priority(location_data.get("cities", []))
                }
            },
            "analysis_timestamp": pd.Timestamp.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Location insights error: {str(e)}")
        return jsonify({"error": f"Failed to generate location insights: {str(e)}"}), 500

def determine_market_type(cities):
    """Determine market type based on cities mentioned."""
    metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Hyderabad']
    tier2_cities = ['Pune', 'Jaipur', 'Ahmedabad', 'Lucknow', 'Kanpur', 'Nagpur']
    
    if any(city in metro_cities for city in cities):
        return "Metro"
    elif any(city in tier2_cities for city in cities):
        return "Tier-2"
    else:
        return "Emerging"

def determine_regional_priority(cities):
    """Determine regional priority based on cities mentioned."""
    if not cities:
        return "National"
    
    # Simplified regional mapping
    for city in cities:
        if city in ['Mumbai', 'Pune']:
            return "West - High Priority"
        elif city in ['Delhi', 'Jaipur']:
            return "North - High Priority"
        elif city in ['Bangalore', 'Chennai']:
            return "South - High Priority"
        elif city in ['Kolkata']:
            return "East - Medium Priority"
    
    return "Regional Focus"


@app.route("/health", methods=["GET"])
def health_check():
    """
    Health check endpoint to verify API and models status.
    """
    return jsonify(
        {
            "status": "healthy",
            "message": "Tata Motors Comment Analysis API is running",
            "models": {
                "bert_loaded": bert_classifier is not None,
                "vader_loaded": vader_analyzer is not None,
                "dataset_loaded": dataset is not None,
                "location_analytics_enabled": True,
            },
            "features": {
                "location_detection": f"{len(INDIAN_CITIES)} cities, {len(INDIAN_STATES)} states",
                "regional_analysis": f"{len(REGIONAL_MAPPING)} regions",
                "geographic_insights": "Enabled"
            }
        }
    )


@app.route("/", methods=["GET"])
def home():
    """
    Root endpoint with API information.
    """
    return jsonify(
        {
            "message": "Tata Motors Customer Comment Analysis API with Location Intelligence",
            "version": "3.0.0",
            "features": [
                "BERT Sentiment Analysis",
                "VADER Rule-based Analysis",
                "Aspect Detection",
                "Strategic Recommendations",
                "Location-based Analytics",
                "Geographic Insights",
                "Regional Performance Tracking",
                "City-wise Sentiment Analysis"
            ],
            "endpoints": {
                "/analyze": "POST - Analyze customer comment with location detection",
                "/location-analytics": "GET - Comprehensive location-based analytics",
                "/city-trends": "GET - City-wise sentiment trends and patterns",
                "/regional-performance": "GET - Regional performance analytics",
                "/location-insights": "POST - Location-specific insights for given text/location",
                "/health": "GET - Health check with model status",
                "/": "GET - API information",
            },
            "new_capabilities": {
                "location_detection": "Automatic detection of cities, states, and regions in comments",
                "geographic_analytics": "City-wise sentiment distribution and engagement metrics",
                "regional_insights": "Regional performance comparisons and market analysis",
                "location_aware_recommendations": "Strategic recommendations enhanced with geographic context"
            },
            "usage": {
                "analyze_endpoint": {
                    "method": "POST",
                    "payload": {"text": "Your customer comment here"},
                    "response": {
                        "input_text": "Original comment",
                        "predicted_category": "Classified category",
                        "identified_aspects": "List of business aspects",
                        "location_analysis": "Detected cities, states, and regions",
                        "sentiment_analysis": {
                            "bert": "ML-based sentiment with confidence",
                            "vader": "Rule-based sentiment with scores",
                        },
                        "strategic_recommendation": "Location-aware actionable recommendations",
                    },
                },
                "location_analytics_endpoint": {
                    "method": "GET",
                    "response": {
                        "city_analytics": "Per-city sentiment and engagement data",
                        "regional_analytics": "Regional performance metrics",
                        "geographic_trends": "Location-based patterns and insights"
                    }
                }
            },
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Tata Motors Comment Analysis Server...")
    print("📊 Initializing ML models and analyzers...")

    # Load dataset
    dataset = load_dataset()
    print(f"✅ Dataset loaded with {len(dataset)} records")

    # Initialize BERT model
    print("🤖 Loading BERT model for ML-based sentiment analysis...")
    bert_classifier = initialize_bert_model()

    # Initialize VADER
    print("📝 Loading VADER for rule-based sentiment analysis...")
    vader_analyzer = initialize_vader()

    # Train intent classifier
    print("🎯 Training intent classification model...")
    intent_classifier = train_intent_classifier(dataset)

    print("✅ All models initialized successfully!")
    print("📡 API endpoints available:")
    print("   POST /analyze - Analyze customer comments with BERT + VADER")
    print("   GET /health - Health check with model status")
    print("   GET / - API information")
    print("🌐 Server starting on http://0.0.0.0:5001")

    # Run the Flask application
    app.run(host="0.0.0.0", port=5001, debug=True)

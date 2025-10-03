# Tata Motors Customer Sentiment Analysis System

A comprehensive Flask backend and interactive frontend for analyzing Tata Motors customer comments using advanced ML (BERT) and rule-based (VADER) sentiment analysis techniques.

## 🚀 Features

### Backend (Flask API)
- **Dual Sentiment Analysis**: BERT (ML-based) + VADER (rule-based)
- **Category Prediction**: Automatically classifies comments into business categories
- **Aspect Detection**: Identifies key business aspects (Service, EV, Features, etc.)
- **Strategic Recommendations**: Generates actionable business insights
- **RESTful API**: Easy integration with any frontend

### Frontend (Interactive Dashboard)
- **Modern UI**: Clean, responsive design with Tata Motors branding
- **Real-time Analysis**: Instant sentiment analysis and recommendations
- **Visual Insights**: Interactive charts and sentiment indicators
- **Sales Strategies**: AI-generated improvement strategies
- **Example Comments**: Quick-start examples for testing

## 📁 Project Structure

```
coep/
├── app.py                              # Flask backend application
├── index.html                          # Frontend dashboard
├── synthetic_tata_motors_data.csv      # Dataset (auto-generated if missing)
└── README.md                           # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Internet connection (for downloading ML models)

### Step 1: Install Dependencies

```bash
pip install flask flask-cors pandas transformers torch vaderSentiment
```

### Step 2: Start the Backend Server

```bash
cd coep
python app.py
```

You should see output like:
```
🚀 Starting Tata Motors Comment Analysis Server...
📊 Initializing ML models and analyzers...
✅ Dataset loaded with X records
🤖 Loading BERT model for ML-based sentiment analysis...
📝 Loading VADER for rule-based sentiment analysis...
✅ All models initialized successfully!
🌐 Server starting on http://0.0.0.0:5000
```

### Step 3: Open the Frontend

Open `index.html` in your web browser or use a local server:

```bash
# Option 1: Open directly in browser
open index.html

# Option 2: Use Python's built-in server
python -m http.server 8000
# Then visit http://localhost:8000
```

## 🎯 Usage

### API Endpoints

#### 1. Analyze Comment
```bash
POST /analyze
Content-Type: application/json

{
  "text": "The service at the dealership was terrible and unprofessional"
}
```

**Response:**
```json
{
  "input_text": "The service at the dealership was terrible and unprofessional",
  "predicted_category": "Complaint / Criticism",
  "identified_aspects": ["Service"],
  "sentiment_analysis": {
    "bert": {
      "sentiment": "negative",
      "confidence": 0.9234
    },
    "vader": {
      "sentiment": "negative",
      "scores": {
        "positive": 0.0,
        "neutral": 0.293,
        "negative": 0.707,
        "compound": -0.8516
      }
    }
  },
  "strategic_recommendation": {
    "insight": "Customers are reporting negative experiences with dealership service...",
    "strategy": "Service Excellence Initiative",
    "action": "Immediately review service center protocols...",
    "priority": "High",
    "sentiment_consensus": "BERT: negative, VADER: negative"
  }
}
```

#### 2. Health Check
```bash
GET /health
```

#### 3. API Information
```bash
GET /
```

### Frontend Dashboard

1. **Enter Comment**: Type or paste customer feedback
2. **Quick Examples**: Click example comments for testing
3. **Analyze**: Get instant AI-powered analysis
4. **Review Results**: See sentiment, category, and aspects
5. **Strategic Insights**: Get actionable business recommendations
6. **Sales Strategies**: View AI-generated improvement strategies

## 🧠 AI Models Used

### BERT (Bidirectional Encoder Representations from Transformers)
- **Model**: `cardiffnlp/twitter-roberta-base-sentiment-latest`
- **Fallback**: `distilbert-base-uncased-finetuned-sst-2-english`
- **Purpose**: Advanced ML-based sentiment classification
- **Output**: Sentiment + confidence score

### VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Rule-based sentiment analysis
- **Purpose**: Lexicon-based sentiment scoring
- **Output**: Detailed sentiment breakdown (positive, neutral, negative, compound)

## 📊 Business Categories

The system classifies comments into:
- **Complaint / Criticism**: Negative feedback requiring immediate attention
- **Praise / Satisfaction**: Positive feedback to leverage
- **Suggestion / Feature Request**: Ideas for product improvement
- **Purchase Intent / Inquiry**: Sales opportunities
- **Competitive Comparison**: Market positioning insights
- **Brand Perception**: Overall brand sentiment

## 🎯 Business Aspects Detected

- **Service**: Dealership, after-sales, customer care
- **Features**: Technology, safety, infotainment
- **EV**: Electric vehicles, charging, battery
- **Price**: Pricing, value, affordability
- **Build Quality**: Construction, materials, durability
- **Performance**: Speed, handling, engine
- **Design**: Styling, appearance, aesthetics
- **Fuel Efficiency**: Mileage, economy
- **Sound System**: Audio, Harman, speakers
- **Comfort**: Seating, space, ergonomics

## 📈 Strategic Recommendations

Based on analysis, the system provides:
- **Business Insights**: What the data reveals
- **Recommended Strategies**: How to respond
- **Immediate Actions**: What to do next
- **Priority Levels**: High/Medium/Low urgency
- **Sentiment Consensus**: Agreement between AI models

## 🚀 Sales & Improvement Strategies

The system generates contextual strategies for:
- **Customer Recovery**: For complaints and negative feedback
- **Advocacy Programs**: For satisfied customers
- **Feature Development**: For suggestions and requests
- **Sales Acceleration**: For purchase inquiries
- **Reputation Management**: For all feedback types

## 🔧 Configuration

### Environment Variables (Optional)
```bash
export FLASK_ENV=development
export FLASK_DEBUG=True
export MODEL_CACHE_DIR=./models
```

### Model Configuration
- Models are downloaded automatically on first run
- GPU acceleration is used if available
- Fallback models ensure reliability

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Install missing dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. **CORS Error**: Ensure Flask server is running on port 5000

3. **Model Download Fails**: Check internet connection, models will download on first use

4. **Memory Issues**: Use CPU-only mode by setting device=-1 in code

### Backend Logs
Check console output for detailed error messages and model loading status.

### Frontend Issues
- Open browser developer tools (F12) to check for JavaScript errors
- Verify API endpoint URLs match your server configuration

## 📱 Browser Compatibility

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## 🚦 Performance

### Backend
- **Cold Start**: 30-60 seconds (model loading)
- **Analysis Time**: 1-3 seconds per comment
- **Memory Usage**: ~2-4GB (with ML models)

### Frontend
- **Load Time**: <2 seconds
- **Analysis Display**: Real-time updates
- **Responsive**: Works on desktop and mobile

## 🔐 Security

- CORS enabled for local development
- Input validation and sanitization
- No sensitive data storage
- Rate limiting recommended for production

## 📝 Development

### Adding New Aspects
Edit the `aspect_keywords` dictionary in `analyze_aspects()` function:

```python
aspect_keywords = {
    "New_Aspect": ["keyword1", "keyword2", "keyword3"],
    # ... existing aspects
}
```

### Custom Models
Replace model names in `initialize_bert_model()`:

```python
model_name = "your-custom-model-name"
```

### Frontend Customization
- Modify CSS variables in `:root` for theming
- Update `generateSalesStrategies()` for custom strategies
- Add new visualizations in the results section

## 📞 Support

For issues, questions, or contributions:
1. Check the troubleshooting section
2. Review console logs for error details
3. Ensure all dependencies are correctly installed
4. Verify backend server is running and accessible

## 🎯 Business Impact

This system helps Tata Motors:
- **Improve Customer Satisfaction** through rapid issue identification
- **Enhance Product Development** via customer feedback analysis
- **Optimize Sales Strategies** with sentiment-driven insights
- **Strengthen Brand Positioning** through competitive intelligence
- **Increase Operational Efficiency** via automated analysis

---

**Built with ❤️ for Tata Motors Customer Excellence**
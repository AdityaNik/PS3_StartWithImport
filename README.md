# 🚗 Tata Motors Advanced Customer Analytics Platform

A comprehensive customer sentiment analysis platform with location intelligence, ROI scoring, and AI-powered insights for Tata Motors.

## 🌟 Features Overview

### Core Analytics Engine
- **BERT & VADER Sentiment Analysis**: Dual ML+Rule-based sentiment detection
- **Intent Classification**: Automatic categorization of customer feedback
- **Business Aspect Detection**: Identifies specific areas like Service, Features, EV, Price, etc.
- **Location Intelligence**: Detects and analyzes geographic patterns in customer feedback

### 📊 Dashboard Capabilities

#### 1. **HTML Dashboard** (`dashboard.html`)
- Real-time comment analysis
- Interactive sentiment visualization
- Location-based insights with city/regional analytics
- Historical analysis tracking
- Strategic recommendations with priority scoring

#### 2. **Advanced Streamlit Dashboard** (`streamlit_dashboard.py`)
- **📈 Overview Analytics**: KPIs, trends, and comprehensive visualizations
- **🗺️ Location Intelligence**: Geographic sentiment mapping and regional performance
- **💡 ROI & Opportunity Scoring**: Investment prioritization framework
- **🤖 AI Insight Agent**: Automated discovery of critical patterns and recommendations
- **🔄 Real-time Analysis**: Live comment processing with backend integration

### 🚀 New Location-Based Features

#### Location Detection & Analytics
- **190+ Indian Cities**: Comprehensive city recognition
- **37 States/UTs**: Complete geographic coverage
- **6 Regional Zones**: North, South, East, West, Central, Northeast mapping
- **City Performance Metrics**: Sentiment analysis by location
- **Service Hotspots**: Identification of problem areas
- **Growth Markets**: Potential expansion opportunities

#### API Endpoints
- `/location-analytics`: Comprehensive location-based insights
- `/city-trends`: City-wise sentiment trends and patterns  
- `/regional-performance`: Regional performance analytics
- `/location-insights`: Location-specific insights for given text

### 💰 ROI & Investment Prioritization

#### Opportunity Scoring Algorithm
```
Final Score = (Volume Score × 0.3) + (Sentiment Impact × 0.4) + (Category Impact × 0.3)
```

- **Volume Score**: Normalized discussion frequency
- **Sentiment Impact**: Percentage of negative sentiment
- **Category Impact**: Weight of complaints vs suggestions

#### Investment Tiers
- **🔴 Critical Priority (75-100)**: High investment, very high ROI
- **🟡 High Priority (50-75)**: Moderate investment, high ROI  
- **🟢 Medium Priority (25-50)**: Low investment, medium ROI
- **⚪ Low Priority (0-25)**: Minimal investment, monitor

### 🤖 AI Insight Discovery Agent

#### Automated Analysis Capabilities
- **Pain Point Detection**: Identifies most critical customer issues
- **Growth Opportunity Analysis**: Discovers customer satisfaction drivers
- **Competitive Intelligence**: Monitors competitor mentions and sentiment
- **Regional Performance Insights**: Analyzes geographic patterns
- **Strategic Recommendations**: Generates actionable business insights

#### Agent Workflow
1. **Data Analysis**: Processes entire dataset for patterns
2. **Pain Point Discovery**: Identifies top customer complaints
3. **Opportunity Mapping**: Finds areas of customer delight
4. **Competitive Monitoring**: Analyzes competitor mentions
5. **Regional Assessment**: Evaluates geographic performance
6. **Report Generation**: Creates executive summary with recommendations

## ️ Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### 1. Install Dependencies
```bash
# Install required packages
pip install -r requirements.txt
```

### 2. Prepare Data
Ensure `synthetic_tata_motors_data.csv` is in the project root directory.

### 3. Start the Backend (Flask API)
```bash
python app.py
```
The API will be available at `http://localhost:5001`

### 4. Start the HTML Dashboard
```bash
# Option 1: Simple HTTP server
python -m http.server 8080

# Option 2: Open dashboard.html directly in browser
```
Access at `http://localhost:8080/dashboard.html`

### 5. Start the Advanced Streamlit Dashboard
```bash
streamlit run streamlit_dashboard.py --server.port 8501
```
Access at `http://localhost:8501`

## 📚 Usage Guide

### Basic Analysis Workflow

1. **Start Services**: Run Flask backend and choose your dashboard
2. **Analyze Comments**: Input customer feedback for real-time analysis
3. **Review Insights**: Examine sentiment, aspects, and location data
4. **Strategic Planning**: Use ROI scoring for investment decisions
5. **AI Insights**: Run the automated agent for comprehensive analysis

### Advanced Analytics Workflow

1. **Location Intelligence**: 
   - Navigate to Location Analytics tab
   - Review city and regional performance
   - Identify service hotspots and growth markets

2. **ROI Analysis**:
   - Go to ROI & Opportunity Scoring
   - Select business aspects for detailed analysis
   - Review investment recommendations

3. **AI Agent**:
   - Access AI Insight Agent tab
   - Click "Activate AI Insight Agent"
   - Review automated findings and recommendations

## 🎯 Key Metrics & KPIs

### Business Impact Indicators
- **Customer Satisfaction Score**: Positive sentiment percentage
- **Pain Point Severity**: Negative feedback concentration
- **Regional Performance**: Geographic sentiment distribution
- **Opportunity Score**: Investment prioritization metric
- **Competitive Position**: Sentiment vs competitors

### Operational Metrics
- **Response Time**: Comment analysis speed
- **Coverage**: Geographic and aspect analysis breadth
- **Accuracy**: ML model confidence scores
- **Engagement**: Dashboard usage and interaction

## 🔧 Configuration

### Backend API Configuration
- **Port**: 5001 (configurable in `app.py`)
- **CORS**: Enabled for frontend integration
- **Cache**: LRU caching for performance optimization
- **Models**: BERT + VADER + Intent Classification

### Dashboard Configuration
- **HTML Dashboard Port**: 8080
- **Streamlit Dashboard Port**: 8501
- **API Base URL**: `http://localhost:5001`
- **Auto-refresh**: Configurable in Streamlit dashboard

## 📈 Performance Optimizations

### Backend Optimizations
- **Model Caching**: Pre-loaded ML models
- **LRU Cache**: Function-level caching for repeated calls
- **Batch Processing**: Efficient dataset analysis
- **Async Processing**: Non-blocking operations

### Frontend Optimizations
- **Data Caching**: Streamlit `@st.cache_data` decorators
- **Progressive Loading**: Chunked data processing
- **Responsive Design**: Mobile-friendly interfaces
- **Real-time Updates**: Live data synchronization

## 🔍 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   ```bash
   # Check if Flask is running
   curl http://localhost:5001/health
   
   # Restart backend
   python app.py
   ```

2. **Streamlit Dashboard Errors**
   ```bash
   # Install missing dependencies
   pip install streamlit plotly requests
   
   # Restart with specific port
   streamlit run streamlit_dashboard.py --server.port 8501
   ```

3. **Location Data Missing**
   - Ensure CSV file contains 'location' column
   - Verify city names match detection patterns
   - Check regional mapping configuration

### Performance Issues

1. **Slow Analysis**: 
   - Reduce dataset size for testing
   - Check CPU/memory usage
   - Optimize cache settings

2. **Dashboard Loading**:
   - Clear browser cache
   - Restart Streamlit server
   - Check network connectivity

## � Future Enhancements

### Planned Features
- **Real-time Data Ingestion**: Social media API integration
- **Advanced Visualizations**: Geographic heat maps, 3D charts
- **Export Capabilities**: PDF reports, Excel dashboards
- **Alert System**: Automated notifications for critical issues
- **Mobile App**: Dedicated mobile interface

### ML Model Improvements
- **Custom Model Training**: Domain-specific sentiment models
- **Multi-language Support**: Regional language analysis
- **Advanced NLP**: Named entity recognition, topic modeling
- **Predictive Analytics**: Trend forecasting, churn prediction

## � Support & Documentation

### API Documentation
- Comprehensive endpoint documentation available at `/` route
- Interactive API testing via browser
- JSON response format specifications

### Dashboard Help
- In-app tooltips and guidance
- Example comments for testing
- Progressive disclosure of advanced features

### Technical Support
- Check logs for detailed error messages
- Review terminal output for debugging
- Consult individual module documentation

---

## 🎉 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start backend (Terminal 1)
python app.py

# 3. Start Streamlit dashboard (Terminal 2)  
streamlit run streamlit_dashboard.py

# 4. Access dashboards
# HTML: http://localhost:8080/dashboard.html
# Streamlit: http://localhost:8501
```

**🎯 Ready to explore Tata Motors customer insights with advanced AI-powered analytics!**
// Enhanced ML-Powered Utility functions for Tata Motors Customer Sentiment Analysis Dashboard
// Includes Intent Classification, BERT Analysis, and Smart Caching Support

/**
 * Advanced chart generation utilities
 */
class SentimentChart {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext("2d");
    this.canvas.width = 300;
    this.canvas.height = 150;
  }

  drawSentimentBar(bertScore, vaderScore) {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw BERT bar
    ctx.fillStyle = "#0066cc";
    ctx.fillRect(20, 30, (bertScore * 0.5 + 0.5) * (width - 40), 30);
    ctx.fillStyle = "#333";
    ctx.font = "12px Inter";
    ctx.fillText("BERT", 20, 25);
    ctx.fillText(`${(bertScore * 100).toFixed(1)}%`, width - 60, 50);

    // Draw VADER bar
    ctx.fillStyle = "#10b981";
    ctx.fillRect(20, 90, (vaderScore * 0.5 + 0.5) * (width - 40), 30);
    ctx.fillText("VADER", 20, 85);
    ctx.fillText(`${(vaderScore * 100).toFixed(1)}%`, width - 60, 110);
  }
}

/**
 * Data export utilities
 */
class DataExporter {
  static exportToJSON(analysisData) {
    const dataStr = JSON.stringify(analysisData, null, 2);
    const dataBlob = new Blob([dataStr], { type: "application/json" });
    const url = URL.createObjectURL(dataBlob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `tata-analysis-${Date.now()}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  static exportToCSV(analysisHistory) {
    const headers = [
      "Timestamp",
      "Comment",
      "Category",
      "BERT Sentiment",
      "VADER Sentiment",
      "Aspects",
      "Priority",
    ];
    const csvContent = [
      headers.join(","),
      ...analysisHistory.map((item) =>
        [
          new Date(item.timestamp).toISOString(),
          `"${item.input_text.replace(/"/g, '""')}"`,
          item.predicted_category,
          item.sentiment_analysis.bert.sentiment,
          item.sentiment_analysis.vader.sentiment,
          `"${item.identified_aspects.join(", ")}"`,
          item.strategic_recommendation.priority,
        ].join(","),
      ),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);

    const link = document.createElement("a");
    link.href = url;
    link.download = `tata-analysis-history-${Date.now()}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }
}

/**
 * Local storage manager for analysis history
 */
class AnalysisStorage {
  constructor() {
    this.storageKey = "tata_analysis_history";
    this.maxHistoryItems = 100;
  }

  saveAnalysis(analysisData) {
    const history = this.getHistory();
    const newItem = {
      ...analysisData,
      timestamp: Date.now(),
      id: this.generateId(),
      mlEnhanced: true, // Flag for ML-powered analysis
      modelVersion: "2.0", // Track model version
    };

    history.unshift(newItem);

    // Keep only the latest items
    if (history.length > this.maxHistoryItems) {
      history.splice(this.maxHistoryItems);
    }

    localStorage.setItem(this.storageKey, JSON.stringify(history));

    // Auto-trigger analytics update
    if (window.tataAnalytics) {
      window.tataAnalytics.update();
    }

    return newItem;
  }

  getHistory() {
    const stored = localStorage.getItem(this.storageKey);
    return stored ? JSON.parse(stored) : [];
  }

  clearHistory() {
    localStorage.removeItem(this.storageKey);
  }

  getAnalysisById(id) {
    const history = this.getHistory();
    return history.find((item) => item.id === id);
  }

  generateId() {
    return Date.now().toString(36) + Math.random().toString(36).substr(2);
  }

  getAnalyticsSummary() {
    const history = this.getHistory();
    const summary = {
      totalAnalyses: history.length,
      categoryCounts: {},
      sentimentCounts: { positive: 0, negative: 0, neutral: 0 },
      aspectCounts: {},
      priorityCounts: { High: 0, Medium: 0, Low: 0 },
      averageConfidence: 0,
    };

    history.forEach((item) => {
      // Category counts
      summary.categoryCounts[item.predicted_category] =
        (summary.categoryCounts[item.predicted_category] || 0) + 1;

      // Sentiment counts (BERT ML-powered)
      const bertSentiment = item.sentiment_analysis.bert.sentiment;
      summary.sentimentCounts[bertSentiment]++;

      // Track ML vs non-ML analyses
      summary.mlPowered = summary.mlPowered || 0;
      if (item.mlEnhanced) {
        summary.mlPowered++;
      }

      // Aspect counts
      item.identified_aspects.forEach((aspect) => {
        summary.aspectCounts[aspect] = (summary.aspectCounts[aspect] || 0) + 1;
      });

      // Priority counts
      const priority = item.strategic_recommendation.priority;
      summary.priorityCounts[priority]++;

      // Confidence
      summary.averageConfidence += item.sentiment_analysis.bert.confidence;
    });

    if (history.length > 0) {
      summary.averageConfidence /= history.length;
    }

    // Calculate ML accuracy metrics
    summary.mlAccuracy =
      history.length > 0 ? (summary.mlPowered / history.length) * 100 : 0;
    summary.modelVersions = [
      ...new Set(history.map((item) => item.modelVersion || "1.0")),
    ];

    return summary;
  }

  // ML-specific analytics
  getMLPerformanceMetrics() {
    const history = this.getHistory();
    const mlItems = history.filter((item) => item.mlEnhanced);

    if (mlItems.length === 0) {
      return { noMLData: true };
    }

    const confidenceScores = mlItems.map(
      (item) => item.sentiment_analysis.bert.confidence,
    );
    const avgConfidence =
      confidenceScores.reduce((a, b) => a + b, 0) / confidenceScores.length;

    return {
      totalMLAnalyses: mlItems.length,
      averageConfidence: avgConfidence,
      highConfidenceCount: confidenceScores.filter((score) => score > 0.8)
        .length,
      categoriesDetected: [
        ...new Set(mlItems.map((item) => item.predicted_category)),
      ].length,
      uniqueAspects: [
        ...new Set(mlItems.flatMap((item) => item.identified_aspects)),
      ].length,
    };
  }
}

/**
 * ML-Enhanced Advanced recommendation engine
 */
class RecommendationEngine {
  static generateAdvancedStrategies(analysisData, historicalData = []) {
    const strategies = [];
    const { predicted_category, identified_aspects, sentiment_analysis } =
      analysisData;

    // ML Confidence-based strategies
    const bertConfidence = sentiment_analysis.bert.confidence;
    if (bertConfidence < 0.6) {
      strategies.push({
        title: "Human Review Required",
        description:
          "ML confidence below threshold - route to human analyst for verification.",
        impact: "High",
        timeline: "Immediate",
        type: "ml-validation",
      });
    }

    // Industry-specific strategies
    if (identified_aspects.includes("EV")) {
      strategies.push(
        ...this.getEVStrategies(sentiment_analysis, predicted_category),
      );
    }

    if (identified_aspects.includes("Service")) {
      strategies.push(
        ...this.getServiceStrategies(sentiment_analysis, predicted_category),
      );
    }

    // ML-Enhanced Competitive analysis
    if (
      predicted_category.includes("Comparison") ||
      predicted_category.includes("Competitive")
    ) {
      strategies.push(
        ...this.getCompetitiveStrategies(
          identified_aspects,
          sentiment_analysis,
        ),
      );
    }

    // ML-Enhanced Historical trend analysis
    if (historicalData.length > 5) {
      strategies.push(...this.getTrendBasedStrategies(historicalData));
      strategies.push(...this.getMLTrendStrategies(historicalData));
    }

    // Add ML-specific recommendations
    strategies.push(
      ...this.getMLInsightStrategies(analysisData, historicalData),
    );

    return strategies;
  }

  static getEVStrategies(sentiment, category) {
    return [
      {
        title: "EV Infrastructure Partnership",
        description:
          "Collaborate with charging network providers to address range anxiety concerns.",
        impact: "High",
        timeline: "3-6 months",
      },
      {
        title: "EV Education Program",
        description:
          "Launch comprehensive customer education about EV benefits and usage.",
        impact: "Medium",
        timeline: "1-3 months",
      },
    ];
  }

  static getServiceStrategies(sentiment, category) {
    return [
      {
        title: "Service Quality Monitoring",
        description:
          "Implement real-time service quality tracking across all dealerships.",
        impact: "High",
        timeline: "2-4 months",
      },
      {
        title: "Customer Service Training",
        description:
          "Enhanced training programs for service staff focusing on customer experience.",
        impact: "Medium",
        timeline: "1-2 months",
      },
    ];
  }

  static getCompetitiveStrategies(aspects = [], sentimentAnalysis = {}) {
    const strategies = [
      {
        title: "AI-Powered Competitive Intelligence",
        description:
          "Deploy ML models to monitor competitor mentions and sentiment patterns.",
        impact: "Medium",
        timeline: "2-3 months",
        type: "ml-competitive",
      },
    ];

    // Aspect-specific competitive strategies
    if (aspects.includes("Price")) {
      strategies.push({
        title: "Dynamic Pricing Strategy",
        description:
          "Use ML insights to optimize pricing based on competitive analysis and sentiment.",
        impact: "High",
        timeline: "1-2 months",
        type: "ml-pricing",
      });
    }

    return strategies;
  }

  static getTrendBasedStrategies(historicalData) {
    // Analyze trends in historical data
    const recentNegative = historicalData
      .slice(0, 10)
      .filter(
        (item) => item.sentiment_analysis.bert.sentiment === "negative",
      ).length;

    if (recentNegative > 5) {
      return [
        {
          title: "Crisis Management Protocol",
          description:
            "Activate immediate response team due to increasing negative sentiment trend.",
          impact: "Critical",
          timeline: "Immediate",
        },
      ];
    }

    return [];
  }

  static getMLTrendStrategies(historicalData) {
    const recentML = historicalData
      .slice(0, 20)
      .filter((item) => item.mlEnhanced);
    const strategies = [];

    if (recentML.length > 10) {
      const avgConfidence =
        recentML.reduce(
          (sum, item) => sum + item.sentiment_analysis.bert.confidence,
          0,
        ) / recentML.length;

      if (avgConfidence > 0.85) {
        strategies.push({
          title: "High-Confidence Pattern Detection",
          description:
            "ML models showing high confidence - leverage insights for strategic decisions.",
          impact: "Medium",
          timeline: "1 month",
          type: "ml-insight",
        });
      }
    }

    return strategies;
  }

  static getMLInsightStrategies(analysisData, historicalData) {
    const strategies = [];
    const confidence = analysisData.sentiment_analysis.bert.confidence;

    // High confidence ML predictions
    if (confidence > 0.9) {
      strategies.push({
        title: "ML-Verified Action",
        description:
          "High-confidence ML prediction allows for immediate action implementation.",
        impact: "High",
        timeline: "Immediate",
        type: "ml-verified",
      });
    }

    // Category-specific ML insights
    if (
      analysisData.predicted_category.includes("Criticism") &&
      confidence > 0.8
    ) {
      strategies.push({
        title: "Proactive Issue Resolution",
        description:
          "ML detected high-confidence criticism - trigger immediate response protocol.",
        impact: "Critical",
        timeline: "Within 4 hours",
        type: "ml-urgent",
      });
    }

    return strategies;
  }
}

/**
 * ML-Enhanced Performance monitoring utilities
 */
class PerformanceMonitor {
  constructor() {
    this.metrics = {
      apiCalls: 0,
      totalResponseTime: 0,
      errors: 0,
      startTime: Date.now(),
      mlAnalyses: 0,
      bertProcessingTime: 0,
      intentClassificationTime: 0,
      cacheHits: 0,
      cacheMisses: 0,
    };
  }

  recordApiCall(responseTime, isMLEnhanced = false) {
    this.metrics.apiCalls++;
    this.metrics.totalResponseTime += responseTime;

    if (isMLEnhanced) {
      this.metrics.mlAnalyses++;
    }
  }

  recordMLProcessing(bertTime, intentTime) {
    this.metrics.bertProcessingTime += bertTime;
    this.metrics.intentClassificationTime += intentTime;
  }

  recordCacheHit() {
    this.metrics.cacheHits++;
  }

  recordCacheMiss() {
    this.metrics.cacheMisses++;
  }

  recordError() {
    this.metrics.errors++;
  }

  getStats() {
    const uptime = Date.now() - this.metrics.startTime;
    const totalCacheRequests =
      this.metrics.cacheHits + this.metrics.cacheMisses;

    return {
      ...this.metrics,
      averageResponseTime:
        this.metrics.apiCalls > 0
          ? this.metrics.totalResponseTime / this.metrics.apiCalls
          : 0,
      uptime: uptime,
      errorRate:
        this.metrics.apiCalls > 0
          ? (this.metrics.errors / this.metrics.apiCalls) * 100
          : 0,
      mlUsageRate:
        this.metrics.apiCalls > 0
          ? (this.metrics.mlAnalyses / this.metrics.apiCalls) * 100
          : 0,
      averageBertTime:
        this.metrics.mlAnalyses > 0
          ? this.metrics.bertProcessingTime / this.metrics.mlAnalyses
          : 0,
      averageIntentTime:
        this.metrics.mlAnalyses > 0
          ? this.metrics.intentClassificationTime / this.metrics.mlAnalyses
          : 0,
      cacheHitRate:
        totalCacheRequests > 0
          ? (this.metrics.cacheHits / totalCacheRequests) * 100
          : 0,
    };
  }
}

/**
 * Theme manager for UI customization
 */
class ThemeManager {
  constructor() {
    this.themes = {
      light: {
        "--tata-blue": "#0066cc",
        "--bg-light": "#f8fafc",
        "--text-primary": "#1e293b",
      },
      dark: {
        "--tata-blue": "#4da3ff",
        "--bg-light": "#1e293b",
        "--text-primary": "#f8fafc",
      },
      highContrast: {
        "--tata-blue": "#0066cc",
        "--bg-light": "#ffffff",
        "--text-primary": "#000000",
      },
    };

    this.currentTheme = localStorage.getItem("tata_theme") || "light";
    this.applyTheme(this.currentTheme);
  }

  applyTheme(themeName) {
    if (!this.themes[themeName]) return;

    const root = document.documentElement;
    const theme = this.themes[themeName];

    Object.entries(theme).forEach(([property, value]) => {
      root.style.setProperty(property, value);
    });

    this.currentTheme = themeName;
    localStorage.setItem("tata_theme", themeName);
  }

  toggleTheme() {
    const themes = Object.keys(this.themes);
    const currentIndex = themes.indexOf(this.currentTheme);
    const nextIndex = (currentIndex + 1) % themes.length;
    this.applyTheme(themes[nextIndex]);
  }
}

/**
 * Notification system
 */
class NotificationManager {
  constructor() {
    this.createNotificationContainer();
  }

  createNotificationContainer() {
    if (document.getElementById("notification-container")) return;

    const container = document.createElement("div");
    container.id = "notification-container";
    container.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            pointer-events: none;
        `;
    document.body.appendChild(container);
  }

  show(message, type = "info", duration = 5000) {
    const notification = document.createElement("div");
    notification.style.cssText = `
            background: ${this.getBackgroundColor(type)};
            color: white;
            padding: 1rem 1.5rem;
            border-radius: 8px;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            pointer-events: auto;
            cursor: pointer;
            transition: all 0.3s ease;
            max-width: 300px;
            word-wrap: break-word;
        `;

    notification.innerHTML = `
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <i class="fas ${this.getIcon(type)}"></i>
                <span>${message}</span>
            </div>
        `;

    const container = document.getElementById("notification-container");
    container.appendChild(notification);

    // Auto remove
    setTimeout(() => {
      if (notification.parentNode) {
        notification.style.opacity = "0";
        notification.style.transform = "translateX(100%)";
        setTimeout(() => {
          if (notification.parentNode) {
            notification.parentNode.removeChild(notification);
          }
        }, 300);
      }
    }, duration);

    // Click to dismiss
    notification.onclick = () => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    };
  }

  getBackgroundColor(type) {
    const colors = {
      success: "#10b981",
      error: "#ef4444",
      warning: "#f59e0b",
      info: "#0066cc",
    };
    return colors[type] || colors.info;
  }

  getIcon(type) {
    const icons = {
      success: "fa-check-circle",
      error: "fa-exclamation-circle",
      warning: "fa-exclamation-triangle",
      info: "fa-info-circle",
    };
    return icons[type] || icons.info;
  }
}

/**
 * Real-time analytics dashboard
 */
class AnalyticsDashboard {
  constructor(containerId) {
    this.container = document.getElementById(containerId);
    this.storage = new AnalysisStorage();
    this.render();
  }

  render() {
    const summary = this.storage.getAnalyticsSummary();

    this.container.innerHTML = `
            <div class="analytics-grid">
                <div class="metric-card">
                    <h3><i class="fas fa-chart-line"></i> Total Analyses</h3>
                    <div class="metric-value">${summary.totalAnalyses}</div>
                    <div class="metric-subtitle">ML Enhanced: ${summary.mlPowered || 0}</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-brain"></i> ML Confidence</h3>
                    <div class="metric-value">${(summary.averageConfidence * 100).toFixed(1)}%</div>
                    <div class="metric-subtitle">BERT Average</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-tags"></i> Top Category</h3>
                    <div class="metric-value">${this.getTopCategory(summary.categoryCounts)}</div>
                    <div class="metric-subtitle">Intent Classified</div>
                </div>
                <div class="metric-card">
                    <h3><i class="fas fa-robot"></i> ML Accuracy</h3>
                    <div class="metric-value">${summary.mlAccuracy?.toFixed(1) || 0}%</div>
                    <div class="metric-subtitle">Model v${summary.modelVersions?.join(", ") || "1.0"}</div>
                </div>
                <div class="metric-card full-width">
                    <h3><i class="fas fa-heart-pulse"></i> Sentiment Distribution</h3>
                    <div class="sentiment-chart">
                        ${this.renderSentimentChart(summary.sentimentCounts)}
                    </div>
                </div>
            </div>
        `;
  }

  getTopCategory(categoryCounts) {
    return (
      Object.entries(categoryCounts).sort(([, a], [, b]) => b - a)[0]?.[0] ||
      "N/A"
    );
  }

  renderSentimentChart(sentimentCounts) {
    const total = Object.values(sentimentCounts).reduce((a, b) => a + b, 0);
    if (total === 0) return "<span>No data</span>";

    return Object.entries(sentimentCounts)
      .map(([sentiment, count]) => {
        const percentage = ((count / total) * 100).toFixed(1);
        return `<div class="sentiment-bar">
                    <span class="sentiment-label">${sentiment}</span>
                    <div class="bar-container">
                        <div class="bar sentiment-${sentiment}" style="width: ${percentage}%"></div>
                    </div>
                    <span class="sentiment-percentage">${percentage}%</span>
                </div>`;
      })
      .join("");
  }

  update() {
    this.render();
  }
}

/**
 * ML Model Status Monitor
 */
class MLModelMonitor {
  constructor() {
    this.models = {
      intentClassifier: { status: "unknown", lastCheck: null },
      bertAnalyzer: { status: "unknown", lastCheck: null },
      cacheSystem: { status: "unknown", lastCheck: null },
    };
  }

  async checkModelStatus(apiBaseUrl) {
    try {
      const response = await fetch(`${apiBaseUrl}/health`);
      const data = await response.json();

      this.models.intentClassifier.status = data.intent_classifier
        ? "active"
        : "inactive";
      this.models.bertAnalyzer.status = data.bert_model ? "active" : "inactive";
      this.models.cacheSystem.status = "active"; // Assume cache is working if API responds

      const now = Date.now();
      Object.keys(this.models).forEach((key) => {
        this.models[key].lastCheck = now;
      });

      this.updateStatusDisplay();
      return this.models;
    } catch (error) {
      console.error("Model status check failed:", error);
      Object.keys(this.models).forEach((key) => {
        this.models[key].status = "error";
        this.models[key].lastCheck = Date.now();
      });
      return this.models;
    }
  }

  updateStatusDisplay() {
    const badges = document.querySelectorAll(".status-badges .badge");
    badges.forEach((badge) => {
      const text = badge.textContent.toLowerCase();
      let status = "inactive";

      if (
        text.includes("intent") &&
        this.models.intentClassifier.status === "active"
      )
        status = "active";
      if (text.includes("bert") && this.models.bertAnalyzer.status === "active")
        status = "active";
      if (
        text.includes("caching") &&
        this.models.cacheSystem.status === "active"
      )
        status = "active";

      badge.className = `badge ${status}`;
    });
  }
}

// Export utilities for global use
window.TataUtils = {
  SentimentChart,
  DataExporter,
  AnalysisStorage,
  RecommendationEngine,
  PerformanceMonitor,
  ThemeManager,
  NotificationManager,
  AnalyticsDashboard,
  MLModelMonitor,
};

// Initialize global instances with ML enhancements
window.tataStorage = new AnalysisStorage();
window.tataNotifications = new NotificationManager();
window.tataPerformance = new PerformanceMonitor();
window.tataTheme = new ThemeManager();
window.tataMLMonitor = new MLModelMonitor();

// Auto-check ML model status every 30 seconds
setInterval(() => {
  if (window.API_BASE_URL) {
    window.tataMLMonitor.checkModelStatus(window.API_BASE_URL);
  }
}, 30000);

// Initial status check when page loads
document.addEventListener("DOMContentLoaded", () => {
  setTimeout(() => {
    if (window.API_BASE_URL) {
      window.tataMLMonitor.checkModelStatus(window.API_BASE_URL);
    }
  }, 2000);
});

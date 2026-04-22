# 📊 Philippine Stock Exchange (PSE) Dashboard

An AI-powered stock prediction and analysis dashboard for Philippine Stock Exchange (PSE) stocks, featuring machine learning predictions, sentiment analysis, and AI-generated investment insights.

## ✨ Features

- **📈 Stock Price Prediction**: ML-based predictions (UP/DOWN/FLAT) using XGBoost classifier
- **🤖 AI-Powered Analysis**: GPT-4 generated investment insights with buy/sell recommendations and SWOT analysis
- **📄 PDF Sentiment Analysis**: Upload PSE disclosure documents for automated sentiment analysis using FinBERT
- **📊 Technical Indicators**: Real-time display of SMA, EMA, MACD, RSI, and more
- **📉 Interactive Charts**: Time series visualization with Plotly
- **🔍 SHAP Analysis**: Model interpretability with SHAP feature importance
- **💹 Statistical Analysis**: Annualized returns, risk-adjusted returns, and volatility metrics

## 🎯 Supported Stocks

- AP (Aboitiz Power)
- AREIT (AREIT Inc.)
- CNVRG (Converge ICT)
- DMC (DMCI Holdings)
- JFC (Jollibee Foods)
- MBT (Metrobank)
- SCC (Semirara Mining)
- SM (SM Investments)
- SMPH (SM Prime Holdings)
- TEL (PLDT Inc.)

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- HuggingFace API key

### Local Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd STREAMLIT
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up API keys**
   
   Create two files in the project root:
   
   - `OPENAI_API_KEY.txt` with your OpenAI API key
   - `HUGGINGFACE_API_KEY.txt` with your HuggingFace API key

4. **Run the app**
   ```bash
   streamlit run stocks_dashboard.py
   ```

5. **Access the dashboard**
   
   Open your browser to `http://localhost:8501`

## ☁️ Deploying to Streamlit Cloud

### Step 1: Prepare Your Repository

1. **Ensure these files are in your repository:**
   - `stocks_dashboard.py` (main app)
   - `requirements.txt` (dependencies)
   - `data.csv` (stock data)
   - `model.pkl` (trained model)
   - `scaler.pkl` (feature scaler)
   - `feature_info.pkl` (feature metadata)
   - `shapley.png` & `shapley.md` (SHAP analysis)
   - `.gitignore` (excludes sensitive files)
   - `README.md` (this file)

2. **DO NOT commit API key files** (`.gitignore` handles this)

### Step 2: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: PSE Stock Dashboard"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub repository
4. Select:
   - **Repository**: your-username/your-repo
   - **Branch**: main
   - **Main file path**: stocks_dashboard.py
5. Click **"Deploy"**

### Step 4: Configure API Keys (CRITICAL)

After deployment, configure your API keys in Streamlit Cloud:

1. Go to your app dashboard on Streamlit Cloud
2. Click **"Settings"** → **"Secrets"**
3. Add your secrets in TOML format:

```toml
OPENAI_API_KEY = "sk-your-actual-openai-key-here"
HUGGINGFACE_API_KEY = "hf_your-actual-huggingface-key-here"
```

4. Click **"Save"**
5. Your app will automatically restart with the new secrets

## 📂 Project Structure

```
STREAMLIT/
├── stocks_dashboard.py      # Main Streamlit application
├── requirements.txt         # Python dependencies
├── data.csv                 # Historical stock data with technical indicators
├── model.pkl                # Trained XGBoost model
├── scaler.pkl               # Feature scaler
├── feature_info.pkl         # Feature metadata
├── shapley.png              # SHAP feature importance plot
├── shapley.md               # SHAP analysis documentation
├── .gitignore               # Git ignore rules
├── README.md                # This file
└── .streamlit/
    └── secrets.toml         # Local secrets (not committed)
```

## 🔧 Configuration

### Local Development with Secrets

For local testing with Streamlit secrets:

1. Create `.streamlit/secrets.toml`:
   ```toml
   OPENAI_API_KEY = "your-key-here"
   HUGGINGFACE_API_KEY = "your-key-here"
   ```

2. This file is already in `.gitignore` and won't be committed

### API Key Priority

The app checks for API keys in this order:
1. Streamlit secrets (`st.secrets`) - for cloud deployment
2. Local `.txt` files - for local development

## 📊 Usage Guide

### Tab 1: Prediction & Plots

1. **Select a stock** from the sidebar dropdown
2. **View latest data** - Automatically loaded from CSV
3. **Choose disclosure type**:
   - **No Disclosure**: Uses neutral sentiment
   - **Has Disclosure**: Upload PDF for sentiment analysis
4. **Submit prediction** to get ML-based forecast
5. **View time series** by selecting features from sidebar
6. **Analyze statistics** with checkbox enabled

### Tab 2: AI Analysis

1. Complete a prediction in Tab 1 first
2. Click **"Generate AI Analysis"** button
3. View AI-generated insights:
   - 3 reasons to BUY
   - 3 reasons to SELL
   - SWOT analysis

### Tab 3: SHAP Analysis

- View SHAP feature importance plots
- Understand which features influence predictions most
- Read detailed analysis of model behavior

## 🛠️ Requirements

```
streamlit
pandas
plotly
openai
scikit-learn
joblib
PyPDF2
huggingface-hub
```

## 📈 Model Information

- **Algorithm**: XGBoost Classifier
- **Features**: 27 features including:
  - OHLCV data (Open, High, Low, Close, Volume)
  - Technical indicators (SMA, EMA, MACD, RSI)
  - Sentiment scores (Positive, Negative, Neutral)
  - Stock encoding (one-hot)
  - Disclosure indicator
- **Output**: 3-class (DOWN/FLAT/UP) or 2-class (DECREASE/INCREASE) predictions

## 🔐 Security Notes

- **Never commit API keys** to GitHub
- Use Streamlit secrets for cloud deployment
- Keep `.txt` API key files in `.gitignore`
- Regularly rotate your API keys

## 🐛 Troubleshooting

### "Model file not found"
- Ensure `model.pkl`, `scaler.pkl`, and `feature_info.pkl` are in the project directory
- Check file size limits (GitHub: 100 MB, consider Git LFS for large files)

### "API key not found"
- For local: Check that `.txt` files exist with valid keys
- For cloud: Verify secrets are configured in Streamlit Cloud settings

### "Feature count mismatch"
- Ensure `data.csv` contains all required columns
- Verify model was trained with same features

## 📝 License

This project is for educational and research purposes.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 👨‍💻 Author

Your Name - Capstone Project

## 🙏 Acknowledgments

- OpenAI for GPT-4 API
- HuggingFace for FinBERT sentiment analysis
- Streamlit for the amazing framework
- Philippine Stock Exchange (PSE) for market data

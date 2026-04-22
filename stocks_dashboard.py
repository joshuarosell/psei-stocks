import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import joblib
from pathlib import Path
from openai import OpenAI
import PyPDF2
from huggingface_hub import InferenceClient
import io

@st.cache_resource
def load_openai_client():
    """Load OpenAI API key and initialize client"""
    try:
        # Check Streamlit secrets first (for cloud deployment)
        if "OPENAI_API_KEY" in st.secrets:
            api_key = st.secrets["OPENAI_API_KEY"]
        else:
            # Fallback to file (for local development)
            api_key_path = Path('OPENAI_API_KEY.txt')
            if api_key_path.exists():
                with open(api_key_path, 'r') as f:
                    api_key = f.read().strip()
            else:
                return None, False
        
        client = OpenAI(api_key=api_key)
        return client, True
    except Exception as e:
        st.error(f"Error loading OpenAI API key: {str(e)}")
        return None, False

@st.cache_resource
def load_huggingface_client():
    """Load HuggingFace API key and initialize client"""
    try:
        # Check Streamlit secrets first
        if "HUGGINGFACE_API_KEY" in st.secrets:
            api_key = st.secrets["HUGGINGFACE_API_KEY"]
        else:
            # Fallback to file
            api_key_path = Path('HUGGINGFACE_API_KEY.txt')
            if api_key_path.exists():
                with open(api_key_path, 'r') as f:
                    api_key = f.read().strip()
                if api_key.startswith('#'):
                    return None, False
            else:
                return None, False
        
        client = InferenceClient(token=api_key)
        return client, True
    except Exception as e:
        st.error(f"Error loading HuggingFace API key: {str(e)}")
        return None, False

@st.cache_resource
def load_model():
    """Load the trained model"""
    model_path = Path('model.pkl')
    
    if model_path.exists():
        try:
            # Load with joblib (for joblib-saved files)
            model = joblib.load(model_path)
            return model, True
        except Exception as e:
            # Fallback to pickle if joblib fails
            try:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                return model, True
            except Exception:
                raise e  # Re-raise the original error
    else:
        return None, False

@st.cache_resource
def load_scaler():
    """Load the feature scaler"""
    scaler_path = Path('scaler.pkl')
    
    if scaler_path.exists():
        try:
            # Load with joblib (for joblib-saved files)
            scaler = joblib.load(scaler_path)
            return scaler, True
        except Exception as e:
            # Fallback to pickle if joblib fails
            try:
                with open(scaler_path, 'rb') as f:
                    scaler = pickle.load(f)
                return scaler, True
            except Exception:
                raise e  # Re-raise the original error
    else:
        return None, False

@st.cache_resource
def load_feature_info():
    """Load feature information for validation"""
    feature_info_path = Path('feature_info.pkl')
    
    if feature_info_path.exists():
        try:
            # Load with joblib (for joblib-saved files)
            feature_info = joblib.load(feature_info_path)
            return feature_info, True
        except Exception as e:
            # Fallback to pickle if joblib fails
            try:
                with open(feature_info_path, 'rb') as f:
                    feature_info = pickle.load(f)
                return feature_info, True
            except Exception:
                raise e  # Re-raise the original error
    else:
        return None, False

@st.cache_data
def load_data():
    # Load your stock data here. For example, you can use a CSV file.
    # Replace 'your_stock_data.csv' with the path to your actual data file.
    data = pd.read_csv('data.csv', index_col='date')
    numeric_columns = data.select_dtypes(include=['float64', 'int64']).columns
    text_columns = data.select_dtypes(include=['object']).columns
    stock_column = 'stock'  # Assuming 'stock' is the column that contains stock names
    unique_stocks = data[stock_column].unique()
    return data, numeric_columns, text_columns, stock_column, unique_stocks

def get_latest_technical_indicators(stock_name, data):
    """
    Extract the most recent technical indicators for a given stock from historical data.
    Returns: dict with keys sma_20, ema_12, ema_26, macd, signal_line, macd_histogram, rsi
    """
    # Filter data for the selected stock
    stock_data = data[data['stock'] == stock_name]
    
    if stock_data.empty:
        # Return neutral/default values if no data found
        return {
            'sma_20': 0.0,
            'ema_12': 0.0,
            'ema_26': 0.0,
            'macd': 0.0,
            'signal_line': 0.0,
            'macd_histogram': 0.0,
            'rsi': 50.0
        }
    
    # Get the most recent row (last row after sorting by date)
    latest_row = stock_data.iloc[-1]
    
    return {
        'sma_20': float(latest_row['sma_20']),
        'ema_12': float(latest_row['ema_12']),
        'ema_26': float(latest_row['ema_26']),
        'macd': float(latest_row['macd']),
        'signal_line': float(latest_row['signal_line']),
        'macd_histogram': float(latest_row['macd_histogram']),
        'rsi': float(latest_row['rsi'])
    }

def get_latest_stock_data(stock_name, data):
    """
    Extract the most recent OHLCV (Open, High, Low, Close, Volume) data for a given stock.
    Returns: dict with keys open, high, low, close, volume, date (or None if no data found)
    """
    # Filter data for the selected stock
    stock_data = data[data['stock'] == stock_name]
    
    if stock_data.empty:
        return None
    
    # Get the most recent row (last row after sorting by date)
    latest_row = stock_data.iloc[-1]
    
    return {
        'open': float(latest_row['open']),
        'high': float(latest_row['high']),
        'low': float(latest_row['low']),
        'close': float(latest_row['close']),
        'volume': int(latest_row['volume']),
        'date': latest_row.name  # Index is the date
    }

def encode_stock_selection(stock_name):
    """
    Convert stock selection to one-hot encoding for model features.
    Returns: dict with keys stock_AP, stock_AREIT, ..., stock_TEL
    """
    # All possible stocks
    stock_list = ['AP', 'AREIT', 'CNVRG', 'DMC', 'JFC', 'MBT', 'SCC', 'SM', 'SMPH', 'TEL']
    
    # Initialize all to 0.0
    encoding = {f'stock_{stock}': 0.0 for stock in stock_list}
    
    # Set selected stock to 1.0
    if stock_name in stock_list:
        encoding[f'stock_{stock_name}'] = 1.0
    
    return encoding

def extract_text_from_pdf(pdf_file):
    """
    Extract text from uploaded PDF file, focusing on Background/Description section.
    Returns: tuple (extracted_text, full_text) or (None, None) if error
    """
    try:
        # Read PDF
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        pdf_text = ""
        for page in pdf_reader.pages:
            pdf_text += page.extract_text()
        
        # Try to extract Background/Description section
        start_marker = "Background/Description of the Disclosure"
        end_marker = "Other Relevant Information"
        
        start_idx = pdf_text.find(start_marker)
        
        if start_idx != -1:
            # Found the section
            end_idx = pdf_text.find(end_marker, start_idx)
            
            if end_idx != -1:
                section_text = pdf_text[start_idx + len(start_marker):end_idx].strip()
            else:
                section_text = pdf_text[start_idx + len(start_marker):].strip()
            
            # Clean the text - remove table indicators
            lines = section_text.split('\n')
            cleaned_lines = []
            
            for line in lines:
                if not line.strip():
                    continue
                # Skip lines with multiple consecutive spaces (common in tables)
                if '  ' in line and any(char.isdigit() for char in line):
                    continue
                cleaned_lines.append(line)
            
            text_to_analyze = ' '.join(cleaned_lines)
            return text_to_analyze[:512], pdf_text
        else:
            # No Background/Description section found - use full text
            return pdf_text[:512], pdf_text
            
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return None, None

def analyze_sentiment_finbert(text, hf_client):
    """
    Analyze sentiment using FinBERT via HuggingFace Inference API.
    Returns: dict with 'positive', 'negative', 'neutral' scores or None if error
    """
    try:
        # Truncate to 512 characters to stay within token limits
        truncated_text = text[:512]
        
        # Call the remote FinBERT model
        result = hf_client.text_classification(
            text=truncated_text,
            model="ProsusAI/finbert"
        )
        
        # Parse results into dict
        sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 0.0}
        
        for item in result:
            label = item['label'].lower()
            score = item['score']
            if label in sentiment_scores:
                sentiment_scores[label] = score
        
        return sentiment_scores
        
    except Exception as e:
        st.error(f"Error analyzing sentiment: {str(e)}")
        return None

def prepare_stock_context(stock_name, tech_indicators, open_price, high_price, low_price, close_price, volume, sent_score, prediction_label, prediction_proba):
    """
    Prepare comprehensive stock context for OpenAI analysis.
    Returns: formatted string with all relevant stock information.
    """
    context = f"""Stock: {stock_name} (Philippine Stock Exchange - PSE)

Latest Price Data:
- Open: ₱{open_price:.2f}
- High: ₱{high_price:.2f}
- Low: ₱{low_price:.2f}
- Close: ₱{close_price:.2f}
- Volume: {volume:,}

Technical Indicators:
- SMA (20): {tech_indicators['sma_20']:.2f}
- EMA (12): {tech_indicators['ema_12']:.2f}
- EMA (26): {tech_indicators['ema_26']:.2f}
- MACD: {tech_indicators['macd']:.4f}
- Signal Line: {tech_indicators['signal_line']:.4f}
- MACD Histogram: {tech_indicators['macd_histogram']:.4f}
- RSI: {tech_indicators['rsi']:.2f}

Sentiment Analysis:
- Sentiment Score: {sent_score:.2f}

ML Model Prediction:
- Predicted Direction: {prediction_label}
- Confidence: {max(prediction_proba):.2%}
"""
    return context

def generate_openai_stock_analysis(client, stock_context):
    """
    Call OpenAI API to generate stock analysis with buy/sell reasons and SWOT.
    Returns: formatted analysis string or error message.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in Philippine Stock Exchange (PSE) stocks. Provide clear, actionable investment insights based on technical indicators and market data. Always consider the Philippine market context and economic conditions."
                },
                {
                    "role": "user",
                    "content": f"""Based on the following stock data from the Philippine Stock Exchange (PSE), provide a comprehensive analysis:

{stock_context}

Please provide:

1. **3 Reasons to BUY** - List 3 compelling reasons why an investor might consider buying this PSE stock
2. **3 Reasons to SELL** - List 3 compelling reasons why an investor might consider selling this PSE stock
3. **SWOT Analysis** - Provide a brief SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for this PSE stock

Format your response clearly with headers and bullet points."""
                }
            ],
            temperature=0.7,
            max_tokens=1500
        )
        
        return response.choices[0].message.content
    
    except Exception as e:
        return f"Error generating analysis: {str(e)}"

def generate_openai_stock_analysis_stream(client, stock_context):
    """
    Call OpenAI API to generate stock analysis with streaming.
    Yields chunks of text as they are generated.
    """
    try:
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst specializing in Philippine Stock Exchange (PSE) stocks. Provide clear, actionable investment insights based on technical indicators and market data. Always consider the Philippine market context and economic conditions."
                },
                {
                    "role": "user",
                    "content": f"""Based on the following stock data from the Philippine Stock Exchange (PSE), provide a comprehensive analysis:

{stock_context}

Please provide:

1. **3 Reasons to BUY** - List 3 compelling reasons why an investor might consider buying this PSE stock
2. **3 Reasons to SELL** - List 3 compelling reasons why an investor might consider selling this PSE stock
3. **SWOT Analysis** - Provide a brief SWOT analysis (Strengths, Weaknesses, Opportunities, Threats) for this PSE stock

Format your response clearly with headers and bullet points."""
                }
            ],
            temperature=0.7,
            max_tokens=1500,
            stream=True  # Enable streaming
        )
        
        # Yield chunks from the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    
    except Exception as e:
        yield f"Error generating analysis: {str(e)}"

df, numeric_columns, text_columns, stock_column, unique_stocks = load_data()

# Load model
try:
    model, model_loaded = load_model()
    if model_loaded:
        st.sidebar.success("✅ Model loaded successfully")
    else:
        st.sidebar.warning("⚠️ Model file not found")
except Exception as e:
    st.sidebar.error(f"❌ Error loading model: {str(e)}")
    model = None
    model_loaded = False

# Load scaler
try:
    scaler, scaler_loaded = load_scaler()
    if scaler_loaded:
        st.sidebar.success("✅ Scaler loaded successfully")
    else:
        st.sidebar.warning("⚠️ Scaler file not found")
except Exception as e:
    st.sidebar.error(f"❌ Error loading scaler: {str(e)}")
    scaler = None
    scaler_loaded = False

# Load feature info
try:
    feature_info, feature_info_loaded = load_feature_info()
    if feature_info_loaded:
        st.sidebar.success("✅ Feature info loaded successfully")
    else:
        st.sidebar.info("ℹ️ Feature info not available (optional)")
except Exception as e:
    st.sidebar.error(f"❌ Error loading feature info: {str(e)}")
    feature_info = None
    feature_info_loaded = False

# Load OpenAI client
try:
    openai_client, openai_loaded = load_openai_client()
    if openai_loaded:
        st.sidebar.success("✅ OpenAI API ready")
    else:
        st.sidebar.warning("⚠️ OpenAI API key not found")
except Exception as e:
    st.sidebar.error(f"❌ Error loading OpenAI: {str(e)}")
    openai_client = None
    openai_loaded = False

# Load HuggingFace client
try:
    hf_client, hf_loaded = load_huggingface_client()
    if hf_loaded:
        st.sidebar.success("✅ HuggingFace API ready")
    else:
        st.sidebar.warning("⚠️ HuggingFace API key not found")
except Exception as e:
    st.sidebar.error(f"❌ Error loading HuggingFace: {str(e)}")
    hf_client = None
    hf_loaded = False

# Stock Selection in Sidebar
st.sidebar.subheader("Time Series Settings")
stock_choice = st.sidebar.selectbox("Stock", options=['AP', 'AREIT', 'CNVRG', 'DMC', 'JFC', 'MBT', 'SCC', 'SM', 'SMPH', 'TEL'])

# Sidebar controls
checkbox = st.sidebar.checkbox("Display the Dataset")
feature_selection = st.sidebar.multiselect(label="Features to plot", options=numeric_columns)

# Initialize session state for prediction results
if 'prediction_data' not in st.session_state:
    st.session_state.prediction_data = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# Initialize widget counter for forcing recreation
if 'widget_key_counter' not in st.session_state:
    st.session_state.widget_key_counter = 0

# Initialize session state for PDF and sentiment
if 'disclosure_type' not in st.session_state:
    st.session_state.disclosure_type = "No Disclosure"
if 'sentiment_scores' not in st.session_state:
    st.session_state.sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
if 'uploaded_pdf_name' not in st.session_state:
    st.session_state.uploaded_pdf_name = None

st.title("📊 Stock Dashboard")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📈 Prediction & Plots", "🤖 AI Analysis", "🔍 SHAP Analysis"])

# ==================== TAB 1: PREDICTION & PLOTS ====================
with tab1:
    st.write("📊 **Stock Prediction** - Latest data auto-loaded from CSV")
    
    # Get latest stock data for selected stock
    latest_data = get_latest_stock_data(stock_choice, df)
    
    if latest_data:
        # Display latest stock data
        st.info(f"**Latest data for {stock_choice}** (Date: {latest_data['date']})")
        
        # Display OHLCV in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Open", f"₱{latest_data['open']:.2f}")
        with col2:
            st.metric("High", f"₱{latest_data['high']:.2f}")
        with col3:
            st.metric("Low", f"₱{latest_data['low']:.2f}")
        with col4:
            st.metric("Close", f"₱{latest_data['close']:.2f}")
        with col5:
            st.metric("Volume", f"{latest_data['volume']:,}")
        
        st.divider()
        
        # Disclosure Selection
        st.write("**Disclosure Settings:**")
        
        # Use counter to create unique keys - forces widget recreation when counter changes
        key_suffix = st.session_state.widget_key_counter
        
        disclosure_type = st.radio(
            "Select Disclosure Type:",
            options=["No Disclosure", "Has Disclosure"],
            key=f"disclosure_radio_{key_suffix}",
            horizontal=True
        )
        
        # Initialize sentiment values
        positive_sent = 0.0
        negative_sent = 0.0
        neutral_sent = 1.0
        
        if disclosure_type == "No Disclosure":
            # Auto-set to neutral sentiment
            st.info("📊 **Sentiment (Auto-set to Neutral)**")
            col_s1, col_s2, col_s3 = st.columns(3)
            with col_s1:
                st.metric("Positive", "0.00%")
            with col_s2:
                st.metric("Negative", "0.00%")
            with col_s3:
                st.metric("Neutral", "100.00%")
            
            # Store in session state
            st.session_state.sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            st.session_state.uploaded_pdf_name = None
        
        else:  # Has Disclosure
            st.write("**Upload Disclosure PDF:**")
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                key=f"pdf_upload_{key_suffix}",
                help="Upload a disclosure PDF document for sentiment analysis"
            )
            
            if uploaded_file is not None:
                # Store filename
                st.session_state.uploaded_pdf_name = uploaded_file.name
                
                with st.spinner("Analyzing PDF..."):
                    # Extract text from PDF
                    extracted_text, full_text = extract_text_from_pdf(uploaded_file)
                    
                    if extracted_text and hf_loaded and hf_client:
                        # Analyze sentiment with FinBERT
                        sentiment_result = analyze_sentiment_finbert(extracted_text, hf_client)
                        
                        if sentiment_result:
                            # Store sentiment scores
                            st.session_state.sentiment_scores = sentiment_result
                            positive_sent = sentiment_result['positive']
                            negative_sent = sentiment_result['negative']
                            neutral_sent = sentiment_result['neutral']
                            
                            # Display extracted text
                            with st.expander("📄 Extracted Text (first 200 chars)"):
                                st.text(extracted_text[:200])
                            
                            # Display sentiment scores
                            st.success("✅ Sentiment Analysis Complete")
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("Positive", f"{positive_sent*100:.2f}%")
                            with col_s2:
                                st.metric("Negative", f"{negative_sent*100:.2f}%")
                            with col_s3:
                                st.metric("Neutral", f"{neutral_sent*100:.2f}%")
                        else:
                            st.error("Failed to analyze sentiment")
                    elif not hf_loaded:
                        st.error("⚠️ HuggingFace API not available. Please add your API key to HUGGINGFACE_API_KEY.txt")
                    else:
                        st.error("Failed to extract text from PDF")
            else:
                # No PDF uploaded yet
                st.info("👆 Please upload a PDF file to analyze sentiment")
                # Use neutral as default
                st.session_state.sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
        
        st.divider()
        
        # Submit and Clear buttons
        col_btn1, col_btn2 = st.columns([1, 1])
        with col_btn1:
            submit_button = st.button("Submit Prediction")
        with col_btn2:
            clear_button = st.button("Clear Results")
        
        # Handle clear button
        if clear_button:
            # Increment counter to force all widgets to recreate with default values
            st.session_state.widget_key_counter += 1
            # Clear prediction data
            st.session_state.prediction_data = None
            st.session_state.analysis_result = None
            st.session_state.sentiment_scores = {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            st.session_state.uploaded_pdf_name = None
            st.rerun()
    else:
        st.error(f"❌ No data found for {stock_choice} in data.csv")
        submit_button = False

    if submit_button and latest_data:
        # Clear previous analysis from session state
        st.session_state.analysis_result = None
        
        # Use auto-retrieved OHLCV data
        open_price = latest_data['open']
        high_price = latest_data['high']
        low_price = latest_data['low']
        close_price = latest_data['close']
        volume = latest_data['volume']
        
        # Get sentiment from session state
        positive_sent = st.session_state.sentiment_scores['positive']
        negative_sent = st.session_state.sentiment_scores['negative']
        neutral_sent = st.session_state.sentiment_scores['neutral']
        
        # Calculate sentiment score: positive_sent - negative_sent
        sent_score = positive_sent - negative_sent
        
        # Map disclosure to binary
        has_disclosure = 1 if disclosure_type == "Has Disclosure" else 0
        
        # Get technical indicators from historical data
        tech_indicators = get_latest_technical_indicators(stock_choice, df)
        
        # Get one-hot encoding for stock selection
        stock_encoding = encode_stock_selection(stock_choice)
        
        # Model prediction
        if model is not None and model_loaded:
            try:
                # Build the 27-element feature array
                features = [
                    open_price, high_price, low_price, close_price, volume,
                    positive_sent, negative_sent, neutral_sent, sent_score, has_disclosure,
                    tech_indicators['sma_20'], tech_indicators['ema_12'], tech_indicators['ema_26'], 
                    tech_indicators['macd'], tech_indicators['signal_line'], tech_indicators['macd_histogram'], 
                    tech_indicators['rsi'],
                    stock_encoding['stock_AP'], stock_encoding['stock_AREIT'], stock_encoding['stock_CNVRG'],
                    stock_encoding['stock_DMC'], stock_encoding['stock_JFC'], stock_encoding['stock_MBT'],
                    stock_encoding['stock_SCC'], stock_encoding['stock_SM'], stock_encoding['stock_SMPH'],
                    stock_encoding['stock_TEL']
                ]
                
                # Validate features if feature_info is available
                if feature_info is not None and feature_info_loaded:
                    expected_features = feature_info.get('feature_names', [])
                    expected_count = len(expected_features) if expected_features else feature_info.get('n_features', 27)
                    
                    if len(features) != expected_count:
                        st.error(f"❌ Feature count mismatch! Expected {expected_count}, got {len(features)}")
                        st.info("Model training used a different number of features. Please check your feature construction.")
                        st.session_state.prediction_data = None
                        st.stop()
                
                # Scale features if scaler is available
                if scaler is not None and scaler_loaded:
                    features_scaled = scaler.transform([features])
                else:
                    features_scaled = [features]
                    st.warning("⚠️ Scaler not available. Using unscaled features.")
                
                # Make prediction
                prediction = model.predict(features_scaled)[0]
                prediction_proba = model.predict_proba(features_scaled)[0]
                
                # Store all data in session state
                st.session_state.prediction_data = {
                    'stock_choice': stock_choice,
                    'open_price': open_price,
                    'high_price': high_price,
                    'low_price': low_price,
                    'close_price': close_price,
                    'volume': volume,
                    'disclosure_type': disclosure_type,
                    'positive_sent': positive_sent,
                    'negative_sent': negative_sent,
                    'neutral_sent': neutral_sent,
                    'sent_score': sent_score,
                    'has_disclosure': has_disclosure,
                    'pdf_filename': st.session_state.uploaded_pdf_name,
                    'tech_indicators': tech_indicators,
                    'prediction': prediction,
                    'prediction_proba': prediction_proba,
                    'num_classes': len(prediction_proba)
                }
                
            except Exception as e:
                st.error(f"❌ Prediction error: {str(e)}")
                st.info("Make sure the input features match the model's training features.")
                st.session_state.prediction_data = None
        else:
            st.warning("⚠️ Model not available. Please ensure model.pkl exists in the directory.")
            st.session_state.prediction_data = None

    # Display prediction results from session state
    if st.session_state.prediction_data:
        data = st.session_state.prediction_data
        
        # Display entered values
        st.success("Input received successfully!")
        input_data = {
        "Stock": data['stock_choice'],
        "Open": data['open_price'],
        "High": data['high_price'],
        "Low": data['low_price'],
        "Close": data['close_price'],
        "Volume": data['volume'],
        "Disclosure Type": data['disclosure_type'],
        "PDF Filename": data.get('pdf_filename', 'N/A'),
        "Positive Sent": data['positive_sent'],
        "Negative Sent": data['negative_sent'],
        "Neutral Sent": data['neutral_sent'],
        "Sent Score": data['sent_score'],
        "Has Disclosure": data['has_disclosure'],
        "SMA 20": data['tech_indicators']['sma_20'],
        "EMA 12": data['tech_indicators']['ema_12'],
        "EMA 26": data['tech_indicators']['ema_26'],
        "MACD": data['tech_indicators']['macd'],
        "Signal Line": data['tech_indicators']['signal_line'],
        "MACD Histogram": data['tech_indicators']['macd_histogram'],
            "RSI": data['tech_indicators']['rsi']
        }
        st.write("**Entered Values:**")
        st.json(input_data)
        
        # Display prediction
        st.subheader("📊 Prediction Result")
        
        if data['num_classes'] == 3:
            # 3-class classification
            labels = ['DOWN', 'FLAT', 'UP']
            result_label = labels[int(data['prediction'])]
            
            if data['prediction'] == 2:
                st.success(f"🔺 **Predicted: {result_label}**")
            elif data['prediction'] == 0:
                st.error(f"🔻 **Predicted: {result_label}**")
            else:
                st.info(f"➡️ **Predicted: {result_label}**")
            
            # Display probabilities
            st.write("**Prediction Probabilities:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("DOWN", f"{data['prediction_proba'][0]:.2%}")
            with col2:
                st.metric("FLAT", f"{data['prediction_proba'][1]:.2%}")
            with col3:
                st.metric("UP", f"{data['prediction_proba'][2]:.2%}")
        else:
            # Binary classification
            labels = ['DECREASE', 'INCREASE']
            result_label = labels[int(data['prediction'])]
            
            if data['prediction'] == 1:
                st.success(f"🔺 **Predicted: {result_label}**")
            else:
                st.error(f"🔻 **Predicted: {result_label}**")
            
            # Display probabilities
            st.write("**Prediction Probabilities:**")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Decrease", f"{data['prediction_proba'][0]:.2%}")
            with col2:
                st.metric("Increase", f"{data['prediction_proba'][1]:.2%}")
    
    # PLOTS SECTION (moved from Tab 2)
    st.divider()
    st.subheader(f"📊 Time Series Analysis for {stock_choice}")
    
    # Filter data for selected stock
    filtered_df = df[df[stock_column] == stock_choice]
    
    # Plot time series data
    if feature_selection:
        fig = px.line(filtered_df, x=filtered_df.index, y=feature_selection, 
                     title=f"{stock_choice} - Selected Features Over Time")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Select features from the sidebar to display time series plots")
    
    # Display dataset if checkbox is selected
    if checkbox:
        st.divider()
        
        # Full dataset
        st.subheader(f"Dataset for {stock_choice}")
        st.dataframe(filtered_df, use_container_width=True)
        
        st.divider()
        
        # Price movements
        st.subheader(f"Price Movements for {stock_choice}")
        st.dataframe(filtered_df[['stock', 'open', 'high', 'low', 'close', 'volume', 'price_change_pct']], 
                    use_container_width=True)
        
        # Calculate and display statistics
        st.divider()
        st.subheader("📊 Statistical Analysis")
        
        col1, col2, col3 = st.columns(3)
        
        # Calculate annualized return (CAGR) based on actual time period
        # Get the first and last dates to calculate actual years
        first_date = pd.to_datetime(filtered_df.index[0])
        last_date = pd.to_datetime(filtered_df.index[-1])
        years = (last_date - first_date).days / 365.25
        
        # Calculate cumulative return: (1 + r1) * (1 + r2) * ... - 1
        cumulative_return = (1 + filtered_df['price_change_pct'] / 100).prod() - 1
        
        # Annualized return (CAGR): (1 + total_return)^(1/years) - 1
        annual_return = ((1 + cumulative_return) ** (1 / years) - 1) * 100
        
        stdev = px.np.std(filtered_df['price_change_pct'] / 100) * px.np.sqrt(252)
        risk_adj_return = annual_return / (stdev * 100)
        
        with col1:
            st.metric("Annual Return", f"{annual_return:.2f}%")
        with col2:
            st.metric("Standard Deviation", f"{stdev * 100:.2f}%")
        with col3:
            st.metric("Risk Adj. Return", f"{risk_adj_return:.4f}")

# ==================== TAB 2: AI ANALYSIS ====================
with tab2:
    st.subheader("🤖 AI-Powered Stock Analysis")
    
    # Check if prediction data exists
    if st.session_state.prediction_data:
        data = st.session_state.prediction_data
        
        # Determine result label based on number of classes
        if data['num_classes'] == 3:
            labels = ['DOWN', 'FLAT', 'UP']
            result_label = labels[int(data['prediction'])]
        else:
            labels = ['DECREASE', 'INCREASE']
            result_label = labels[int(data['prediction'])]
        
        if openai_loaded and openai_client:
            if st.button("🤖 Generate AI Analysis", key=f"ai_analysis_{data['num_classes']}class"):
                # Prepare stock context
                stock_context = prepare_stock_context(
                    stock_name=data['stock_choice'],
                    tech_indicators=data['tech_indicators'],
                    open_price=data['open_price'],
                    high_price=data['high_price'],
                    low_price=data['low_price'],
                    close_price=data['close_price'],
                    volume=data['volume'],
                    sent_score=data['sent_score'],
                    prediction_label=result_label,
                    prediction_proba=data['prediction_proba']
                )
                
                # Stream the response using st.write_stream
                stream = generate_openai_stock_analysis_stream(openai_client, stock_context)
                full_response = st.write_stream(stream)
                
                # Store the complete response in session state
                st.session_state.analysis_result = full_response
            
            # Display analysis if it exists in session state (and button wasn't just clicked)
            elif st.session_state.analysis_result:
                st.markdown(st.session_state.analysis_result)
        else:
            st.info("💡 OpenAI API not available. Add your API key to OPENAI_API_KEY.txt to enable AI-powered analysis.")
    else:
        st.info("👈 Please submit a prediction in the 'Prediction & Plots' tab first to generate AI analysis.")

# ==================== TAB 3: SHAP ANALYSIS ====================
with tab3:
    # st.header("🔍 SHAP Feature Importance Analysis")
    st.subheader("Understanding Model Predictions")

    # Check if SHAP files exist
    shap_img_path = Path('shapley.png')
    shap_md_path = Path('shapley.md')
    
    if shap_img_path.exists() and shap_md_path.exists():
        # Display SHAP plot
        st.image('shapley.png', caption='SHAP Feature Importance - XGBoost (All Classes)', 
                use_container_width=True)
        
        st.divider()
        
        # Load and display SHAP explanation markdown
        with open(shap_md_path, 'r', encoding='utf-8') as f:
            shap_text = f.read()
        
        # Display the SHAP analysis
        st.markdown("### 📊 Detailed Analysis")
        st.markdown(shap_text)
        
    else:
        st.warning("⚠️ SHAP analysis files (shapley.png, shapley.md) not found. Please ensure they are in the project directory.")
        
        if not shap_img_path.exists():
            st.info("Missing: shapley.png")
        if not shap_md_path.exists():
            st.info("Missing: shapley.md")
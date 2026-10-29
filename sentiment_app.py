import io
import pickle
import sys
from datetime import datetime
import numpy as np
import streamlit as st

# Compatibility bridge for TensorFlow / Keras 2 & 3
try:
    import tf_keras

    sys.modules.setdefault("keras.preprocessing", tf_keras.preprocessing)
    sys.modules.setdefault("keras.preprocessing.text", tf_keras.preprocessing.text)
    sys.modules.setdefault(
        "keras.preprocessing.sequence", tf_keras.preprocessing.sequence
    )
    load_keras_model = tf_keras.models.load_model
    pad_sequences = tf_keras.preprocessing.sequence.pad_sequences
except ImportError:
    import tensorflow as tf

    load_keras_model = tf.keras.models.load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences

from text_sources import (
    MAX_CSV_BYTES,
    MAX_CSV_ROWS,
    read_texts_from_csv,
    render_batch_results_csv,
)

# Page Configuration
st.set_page_config(
    page_title="NeuralSense | Deep Learning Sentiment Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Hero Section */
.hero-container {
    background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #0f172a 100%);
    border: 1px solid rgba(139, 92, 246, 0.3);
    border-radius: 20px;
    padding: 36px 32px;
    margin-bottom: 28px;
    box-shadow: 0 16px 36px -10px rgba(0, 0, 0, 0.35);
    color: #ffffff;
    position: relative;
    overflow: hidden;
}

.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    margin: 0;
    background: linear-gradient(120deg, #ffffff 30%, #c4b5fd 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.5px;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #cbd5e1;
    margin-top: 10px;
    margin-bottom: 20px;
    max-width: 780px;
    line-height: 1.6;
}

.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

.tag-badge {
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.18);
    color: #e2e8f0;
    padding: 6px 14px;
    border-radius: 30px;
    font-size: 0.82rem;
    font-weight: 600;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* Result Cards */
.result-card {
    border-radius: 16px;
    padding: 24px;
    margin: 18px 0;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.2);
    border: 1px solid;
    animation: fadeIn 0.4s ease-out;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-card.positive {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.12) 0%, rgba(5, 150, 105, 0.05) 100%);
    border-color: rgba(16, 185, 129, 0.4);
}

.result-card.negative {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.12) 0%, rgba(220, 38, 38, 0.05) 100%);
    border-color: rgba(239, 68, 68, 0.4);
}

.sentiment-label-pos {
    color: #10b981;
    font-size: 1.6rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 10px;
}

.sentiment-label-neg {
    color: #ef4444;
    font-size: 1.6rem;
    font-weight: 800;
    display: flex;
    align-items: center;
    gap: 10px;
}

.metric-box {
    background: rgba(255, 255, 255, 0.04);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

.metric-num {
    font-size: 1.5rem;
    font-weight: 700;
    color: #f8fafc;
}

.metric-title {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
    margin-top: 4px;
}

/* History Card */
.history-item {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 12px;
}

/* Sample preset pills */
.preset-btn {
    font-size: 0.85rem !important;
}

/* Project Footer */
.footer-note {
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
    padding: 30px 0 15px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    margin-top: 40px;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# Cache model and tokenizer loading
@st.cache_resource(show_spinner=False)
def load_assets():
    try:
        with open("tokenizer.pkl", "rb") as handle:
            tokenizer = pickle.load(handle)
        model = load_keras_model("model.h5")
        return tokenizer, model, None
    except Exception as e:
        return None, None, str(e)


def predict_sentiment(review_text: str, tokenizer, model):
    sequence = tokenizer.texts_to_sequences([review_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded_sequence, verbose=0)
    score = float(prediction[0][0])
    is_positive = score > 0.5
    sentiment = "Positive" if is_positive else "Negative"
    confidence = score if is_positive else (1.0 - score)
    return sentiment, confidence, score


def render_hero():
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">🧠 Sentiment Analysis</h1>
            <p class="hero-subtitle">
                An end-to-end Natural Language Processing system utilizing a Bidirectional LSTM Neural Network trained on 50,000 IMDB movie reviews to classify semantic polarity and emotional nuance in real time.
            </p>
            <div class="badge-row">
                <span class="tag-badge">⚡ Architecture: LSTM + Word Embeddings</span>
                <span class="tag-badge">📚 Vocab Size: 5,000 Words</span>
                <span class="tag-badge">⏱ Sequence Length: 200 Tokens</span>
                <span class="tag-badge">🎯 Benchmark Accuracy: ~88.5%</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def single_prediction_tab(tokenizer, model):
    st.markdown("### ✍️ Input Text Analysis")

    # Sample Buttons
    st.markdown("**Quick Test Samples:**")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)

    sample_to_use = None
    if col_s1.button("🌟 Positive Masterpiece"):
        sample_to_use = "An absolute cinematic masterpiece! The emotional depth, breathtaking cinematography, and stellar performances made this an unforgettable experience."
    if col_s2.button("⚠️ Critically Flawed"):
        sample_to_use = "A colossal waste of potential. The screenplay was messy, dialogues felt forced, and the pacing was completely unengaging."
    if col_s3.button("⚖️ Mixed / Nuanced"):
        sample_to_use = "The visual effects were undeniably gorgeous and sound design was top notch, but the story felt predictable and lacked emotional weight."
    if col_s4.button("🎬 Classic Thriller"):
        sample_to_use = "A gripping psychological thriller that kept me on the edge of my seat from start to finish. Highly recommended!"

    if "current_input" not in st.session_state:
        st.session_state.current_input = ""

    if sample_to_use:
        st.session_state.current_input = sample_to_use

    with st.form("sentiment_form"):
        user_text = st.text_area(
            "Enter movie review or user feedback:",
            value=st.session_state.current_input,
            placeholder="Type or paste any movie review, audience critique, or sentiment text here...",
            height=140,
        )
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            submit_btn = st.form_submit_button(
                "🔍 Analyze Sentiment", use_container_width=True
            )

    if submit_btn and user_text.strip():
        with st.spinner("Processing through neural network layers..."):
            sentiment, confidence, raw_score = predict_sentiment(
                user_text, tokenizer, model
            )

            word_count = len(user_text.split())
            char_count = len(user_text)

            # Store in session history
            if "history" not in st.session_state:
                st.session_state.history = []

            timestamp_str = datetime.now().strftime("%b %d, %Y - %I:%M:%S %p")
            st.session_state.history.insert(
                0,
                {
                    "text": user_text,
                    "sentiment": sentiment,
                    "confidence": f"{confidence:.1%}",
                    "raw_score": f"{raw_score:.4f}",
                    "timestamp": timestamp_str,
                },
            )

            # Display styled Result Card
            is_pos = sentiment == "Positive"
            card_class = "positive" if is_pos else "negative"
            icon = "✨" if is_pos else "⚡"
            label_class = "sentiment-label-pos" if is_pos else "sentiment-label-neg"

            st.markdown(
                f"""
                <div class="result-card {card_class}">
                    <div class="{label_class}">
                        <span>{icon} Classified Sentiment: {sentiment.upper()}</span>
                    </div>
                    <p style="color: #cbd5e1; margin-top: 6px; font-size: 0.95rem;">
                        The model predicted this review with <b>{confidence:.2%} confidence</b> (Raw sigmoid activation: {raw_score:.4f}).
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Metrics Row
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-num">{sentiment}</div>
                        <div class="metric-title">Verdict</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-num">{confidence:.1%}</div>
                        <div class="metric-title">Confidence</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m3:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-num">{word_count}</div>
                        <div class="metric-title">Words</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with m4:
                st.markdown(
                    f"""
                    <div class="metric-box">
                        <div class="metric-num">{char_count}</div>
                        <div class="metric-title">Characters</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Visual Confidence Progress Bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.write("**Confidence Calibration Gauge:**")
            st.progress(float(confidence))


def batch_prediction_tab(tokenizer, model):
    st.markdown("### 📂 Bulk CSV Dataset Processor")
    st.write(
        "Upload a CSV file containing reviews, tweets, or feedback to perform automated batch sentiment classification."
    )

    uploaded_file = st.file_uploader(
        "Choose a CSV file (supported columns: review, text, comment, message, Tweet Text)",
        type=["csv"],
        key="csv_batch_uploader",
    )

    if uploaded_file is not None:
        if uploaded_file.size > MAX_CSV_BYTES:
            st.error("Uploaded CSV file exceeds the 5 MiB size limit.")
            return

        try:
            batch = read_texts_from_csv(uploaded_file.getvalue())
        except ValueError as err:
            st.error(f"Error parsing CSV: {str(err)}")
            return

        if batch.truncated:
            st.info(
                f"ℹ️ For performance and responsiveness, the first {MAX_CSV_ROWS} rows are analyzed."
            )

        results = []
        pos_count = 0
        neg_count = 0

        progress_bar = st.progress(0)
        total_items = len(batch.texts)

        for idx, text_item in enumerate(batch.texts):
            sent, conf, _ = predict_sentiment(text_item, tokenizer, model)
            if sent == "Positive":
                pos_count += 1
            else:
                neg_count += 1
            results.append(
                {
                    "Index": idx + 1,
                    "Text": text_item,
                    "Sentiment": sent,
                    "Confidence": f"{conf:.1%}",
                }
            )
            progress_bar.progress((idx + 1) / total_items)

        st.success(
            f"Successfully processed {len(results)} rows from column `{batch.column}`!"
        )

        # Batch summary metrics
        b1, b2, b3 = st.columns(3)
        b1.metric("Total Analyzed", len(results))
        b2.metric("Positive Sentiment", f"{pos_count} ({pos_count/len(results):.1%})")
        b3.metric("Negative Sentiment", f"{neg_count} ({neg_count/len(results):.1%})")

        st.dataframe(results, use_container_width=True)

        csv_download = render_batch_results_csv(
            [
                {
                    "review": r["Text"],
                    "sentiment": r["Sentiment"],
                    "confidence": r["Confidence"],
                }
                for r in results
            ]
        )

        st.download_button(
            label="📥 Download Batch Results (CSV)",
            data=csv_download,
            file_name="sentiment_batch_results.csv",
            mime="text/csv",
            key="download_batch_csv",
        )


def history_tab():
    st.markdown("### 📋 Session History")

    if "history" not in st.session_state or not st.session_state.history:
        st.info("No reviews analyzed in this session yet. Try analyzing a review!")
        return

    col_h1, col_h2 = st.columns([1, 4])
    with col_h1:
        if st.button("🗑️ Clear History", key="btn_clear_hist"):
            st.session_state.history = []
            st.rerun()

    for item in st.session_state.history:
        is_pos = item["sentiment"] == "Positive"
        badge = "🟢 Positive" if is_pos else "🔴 Negative"
        st.markdown(
            f"""
            <div class="history-item">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                    <span style="font-weight:700; font-size:1.05rem;">{badge} ({item['confidence']})</span>
                    <span style="color:#94a3b8; font-size:0.8rem;">{item['timestamp']}</span>
                </div>
                <div style="color:#e2e8f0; font-size:0.95rem; line-height:1.5;">{item['text']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Format Markdown for download
    md_buffer = io.StringIO()
    md_buffer.write("# Sentiment Analysis Session Report\n\n")
    for item in st.session_state.history:
        md_buffer.write(f"### Review Analyzed ({item['timestamp']})\n")
        md_buffer.write(f"**Text**: {item['text']}\n\n")
        md_buffer.write(f"- **Verdict**: {item['sentiment']}\n")
        md_buffer.write(f"- **Confidence**: {item['confidence']}\n")
        md_buffer.write(f"- **Raw Score**: {item['raw_score']}\n\n---\n\n")

    st.download_button(
        label="📥 Export History (Markdown)",
        data=md_buffer.getvalue(),
        file_name="sentiment_session_history.md",
        mime="text/markdown",
        key="btn_dl_hist_md",
    )


def main():
    render_hero()

    tokenizer, model, error = load_assets()

    if error:
        st.error(
            f"⚠️ Could not load neural network weights or tokenizer dictionary: {error}"
        )
        st.info(
            "Please verify that `model.h5` and `tokenizer.pkl` are located in the project directory."
        )
        return

    # Modern Navigation Tabs
    tab1, tab2, tab3 = st.tabs(
        [
            "🔍 Single Review Analyzer",
            "📂 Batch CSV Processor",
            "📋 History & Analytics",
        ]
    )

    with tab1:
        single_prediction_tab(tokenizer, model)
    with tab2:
        batch_prediction_tab(tokenizer, model)
    with tab3:
        history_tab()

    # Academic Project Footer
    st.markdown(
        """
        <div class="footer-note">
            NLP & Deep Learning Project | Sentiment Classification Engine using Recurrent Neural Networks (LSTM)
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

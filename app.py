import streamlit as st
import torch
import pandas as pd
import altair as alt
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    pipeline,
)

# --- 1. Model Configuration ---

MODEL_PATH = "./sentiment_model_finetuned"
SENTIMENT_MAP = {0: "NEGATIVE 🔴", 1: "NEUTRAL 🟡", 2: "POSITIVE 🟢"}
SENTIMENT_MAP_COLORS = {
    "NEGATIVE 🔴": "#FF6347",
    "NEUTRAL 🟡": "#FFD700",
    "POSITIVE 🟢": "#3CB371",
}

# The 5 Standard Test Examples (English)
TEST_EXAMPLES = [
    {
        "text": "X airline is the best, excellent flight, wonderful services!",
        "sentiment": "POSITIVE",
    },
    {
        "text": "I've been waiting for 5 hours and I get no information, absolutely unacceptable.",
        "sentiment": "NEGATIVE",
    },
    {
        "text": "I received a confirmation email for my booking, the flight is on time.",
        "sentiment": "NEUTRAL",
    },
    {
        "text": "The staff was rude and ruined the trip. I will never fly with them again!",
        "sentiment": "NEGATIVE",
    },
    {
        "text": "I am neutral about the seats, but the takeoff punctuality was perfect.",
        "sentiment": "NEUTRAL",
    },
]

# Failure Cases for Qualitative Analysis (Requirement Stage 4)
# Focusing on the state before/without fine-tuning
ERROR_EXAMPLES = [
    {
        "text": "The flight was terrible... Just kidding, it was excellent!",
        "real_sentiment": "POSITIVE 🟢",
        "before_training": "NEGATIVE 🔴",
        "cause": "The model is initially fooled by the keyword 'terrible'. It lacks the ability to process the irony and the reversal at the end of the sentence.",
    },
    {
        "text": "The food wasn't bad, it was actually okay.",
        "real_sentiment": "NEUTRAL 🟡",
        "before_training": "NEGATIVE 🔴",
        "cause": "Without specific training, the model focuses only on the word 'bad' and fails to correctly interpret the negation 'wasn't bad'.",
    },
    {
        "text": "I can't believe how late the flight was. Horrible experience!",
        "real_sentiment": "NEGATIVE 🔴",
        "before_training": "NEUTRAL 🟡",
        "cause": "Standard models often miss the intensity of adjectives like 'horrible' when embedded in complex phrases like 'I can't believe'.",
    },
]


@st.cache_resource
def load_model_and_pipeline():
    """Loads the fine-tuned model and tokenizer from the local path."""
    try:
        tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
        model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            return_all_scores=True,
            device=(0 if torch.cuda.is_available() else -1),
        )
        return sentiment_pipeline
    except Exception as e:
        st.error(f"Error loading model: {e}. Ensure '{MODEL_PATH}' folder exists.")
        st.stop()


# Initialize the inference pipeline
pipeline_loaded = load_model_and_pipeline()


def analyze_sentiment(text):
    if not text.strip():
        return None
    result = pipeline_loaded(text)
    scores = result[0]
    max_score = 0.0
    predicted_label_index = 0

    for score_info in scores:
        label_index = int(score_info["label"].split("_")[-1])
        score = score_info["score"]
        if score > max_score:
            max_score = score
            predicted_label_index = label_index

    return SENTIMENT_MAP.get(predicted_label_index, "UNKNOWN"), max_score, scores


# --- 2. Streamlit Interface ---

st.set_page_config(page_title="US Airline Sentiment Dashboard", layout="centered")

st.title("✈️ Airline Sentiment Analysis Dashboard")
st.subheader("DistilBERT Model Fine-tuned on Twitter Data")

# Performance Metrics Header
st.markdown("---")
st.markdown("**Current Model Performance (Final Validation):**")
st.markdown(f"**Accuracy:** `{0.8309}` | **F1 Score (Weighted):** `{0.8300}`")
st.markdown("---")

# Section 1: Live Testing
st.markdown("### 1. Real-time Analysis")
user_input = st.text_area(
    "Enter a tweet or customer feedback in English:",
    placeholder="Ex: My flight was delayed but the cabin crew was very helpful.",
    height=100,
)

if st.button("Run Analysis"):
    if user_input:
        with st.spinner("Analyzing text..."):
            sentiment, score, all_scores = analyze_sentiment(user_input)
            st.success(f"### Detected Sentiment: {sentiment}")
            st.write(f"**Confidence Level:** {score:.2%}")

            # Visualization of probabilities
            data = {
                "Sentiment": [
                    SENTIMENT_MAP[int(s["label"].split("_")[-1])] for s in all_scores
                ],
                "Probability": [s["score"] for s in all_scores],
            }
            chart = (
                alt.Chart(pd.DataFrame(data))
                .mark_bar()
                .encode(
                    x=alt.X("Sentiment", sort=list(SENTIMENT_MAP.values())),
                    y="Probability",
                    color=alt.Color(
                        "Sentiment",
                        scale=alt.Scale(
                            domain=list(SENTIMENT_MAP.values()),
                            range=list(SENTIMENT_MAP_COLORS.values()),
                        ),
                    ),
                )
                .properties(height=300)
            )
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Please provide an input text.")

# Section 2: Fixed Examples Demo
st.markdown("---")
st.markdown("### 2. Standard Benchmark Examples")
results_5 = []
for ex in TEST_EXAMPLES:
    pred, conf, _ = analyze_sentiment(ex["text"])
    results_5.append(
        {
            "Customer Tweet": ex["text"],
            "Target Label": ex["sentiment"],
            "Model Prediction": pred,
            "Confidence": f"{conf:.2%}",
        }
    )
st.dataframe(pd.DataFrame(results_5), use_container_width=True, hide_index=True)

# Section 3: Model Failure Cases (Before Improvement)
st.markdown("---")
st.markdown("### 3. Qualitative Analysis: Failure Cases")
st.markdown(
    "This section illustrates how the model would react **before** the final improvements or the typical errors of a base model."
)


for i, err_ex in enumerate(ERROR_EXAMPLES):
    with st.expander(f"Failure Case #{i + 1}: Complex Context"):
        st.write(f"**Input Text:** *\"{err_ex['text']}\"*")
        st.write(f"**Real Sentiment:** {err_ex['real_sentiment']}")
        st.error(
            f"**Predicted Sentiment (Before Improvement):** {err_ex['before_training']}"
        )
        st.info(f"**Probable Cause of Error:** {err_ex['cause']}")

# Section 4: Training Context
st.markdown("---")
st.markdown("### 4. Training Data Distribution")
try:
    df_dist = pd.read_csv("sentiment_distribution.csv")
    dist_chart = (
        alt.Chart(df_dist)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment", sort=["negative", "neutral", "positive"]),
            y=alt.Y("Procent", title="Percentage (%)"),
            color=alt.Color(
                "Sentiment",
                scale=alt.Scale(
                    domain=["negative", "neutral", "positive"],
                    range=["#FF6347", "#FFD700", "#3CB371"],
                ),
            ),
        )
        .properties(title="Sentiment Balance in Dataset")
    )
    st.altair_chart(dist_chart, use_container_width=True)
except FileNotFoundError:
    st.warning("Training distribution file not found.")

st.markdown(
    """
<div style="background-color: #262730; padding: 15px; border-radius: 10px; margin-top: 20px; border: 1px solid #444;">
    <strong>Technical Summary:</strong> The "Failure Cases" highlight the limitations of standard NLP models when facing sarcasm, double negations, or high-intensity emotions.
</div>
""",
    unsafe_allow_html=True,
)

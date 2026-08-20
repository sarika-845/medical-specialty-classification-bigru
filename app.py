import streamlit as st
import numpy as np
import pickle
import os
import tensorflow as tf
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Medical Specialty Classifier",
    page_icon="🏥",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main {
    background-color: #0e1117;
}

.title {
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #a8b0bd;
    margin-bottom: 35px;
}

.result-card {
    background: linear-gradient(135deg, #182235, #111827);
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 30px;
    margin-top: 25px;
    margin-bottom: 25px;
}

.result-label {
    color: #9ca3af;
    font-size: 16px;
    font-weight: 600;
}

.result-value {
    color: #60a5fa;
    font-size: 34px;
    font-weight: 800;
    margin-top: 8px;
}

.confidence-value {
    color: #22c55e;
    font-size: 28px;
    font-weight: 700;
    margin-top: 8px;
}

.footer {
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #374151;
    color: #9ca3af;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():

    artifacts_folder = "artifacts"

    model_path = os.path.join(
        artifacts_folder,
        "medical_specialty_bigru.keras"
    )

    tokenizer_path = os.path.join(
        artifacts_folder,
        "tokenizer.pkl"
    )

    label_encoder_path = os.path.join(
        artifacts_folder,
        "label_encoder.pkl"
    )

    config_path = os.path.join(
        artifacts_folder,
        "model_config.pkl"
    )

    # Check files
    required_files = [
        model_path,
        tokenizer_path,
        label_encoder_path,
        config_path
    ]

    for file in required_files:
        if not os.path.exists(file):
            raise FileNotFoundError(
                f"Missing artifact: {file}"
            )

    # Load model
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )

    # Load tokenizer
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)

    # Load label encoder
    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Load configuration
    with open(config_path, "rb") as f:
        model_config = pickle.load(f)

    return model, tokenizer, label_encoder, model_config


# ============================================================
# LOAD MODEL
# ============================================================

try:

    model, tokenizer, label_encoder, model_config = load_artifacts()

except Exception as e:

    st.error("Unable to load the trained model or artifacts.")

    st.exception(e)

    st.stop()


# ============================================================
# GET MAX LENGTH
# ============================================================

if isinstance(model_config, dict):

    max_length = model_config.get(
        "max_length",
        model_config.get(
            "max_len",
            500
        )
    )

else:

    max_length = 500


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_specialty(text):

    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])

    # Pad sequence
    padded_sequence = tf.keras.utils.pad_sequences(
        sequence,
        maxlen=max_length,
        padding="post",
        truncating="post"
    )

    # Prediction
    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )[0]

    # Highest probability class
    predicted_index = np.argmax(probabilities)

    # Convert number back to specialty
    predicted_specialty = label_encoder.inverse_transform(
        [predicted_index]
    )[0]

    # Confidence
    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_specialty,
        confidence,
        probabilities
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="title">🏥 Medical Specialty Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered classification of medical transcriptions '
    'using a Bidirectional GRU neural network.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# INPUT
# ============================================================

st.subheader("📝 Enter Medical Transcription")

st.write(
    "Paste a clinical transcription below and the model "
    "will predict the most likely medical specialty."
)

sample_text = st.text_area(
    "Medical Transcription",
    height=220,
    placeholder="""Example:

The patient underwent a portable chest X-ray for evaluation
of shortness of breath. The cardiac silhouette is within
normal limits. No focal air-space consolidation, pleural
effusion, or pneumothorax is identified.""",
    label_visibility="collapsed"
)


# ============================================================
# PREDICT BUTTON
# ============================================================

predict_button = st.button(
    "🔍 Predict Medical Specialty",
    use_container_width=True,
    type="primary"
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    if not sample_text.strip():

        st.warning(
            "Please enter a medical transcription first."
        )

    else:

        with st.spinner("Analyzing medical transcription..."):

            prediction, confidence, probabilities = predict_specialty(
                sample_text
            )

        # ----------------------------------------------------
        # RESULT CARD
        # ----------------------------------------------------

        st.markdown(
            '<div class="result-card">',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="result-label">'
                'Predicted Medical Specialty'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="result-value">'
                f'{prediction}'
                f'</div>',
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="result-label">'
                'Prediction Confidence'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f'<div class="confidence-value">'
                f'{confidence * 100:.2f}%'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )


        # ----------------------------------------------------
        # PROBABILITY DISTRIBUTION
        # ----------------------------------------------------

        st.subheader("📊 Prediction Probability Distribution")

        class_names = label_encoder.classes_

        probability_percentages = probabilities * 100

        fig = go.Figure()

        fig.add_trace(
            go.Bar(
                x=class_names,
                y=probability_percentages,
                text=[
                    f"{p:.2f}%"
                    for p in probability_percentages
                ],
                textposition="auto"
            )
        )

        fig.update_layout(
            title="Model Confidence Across Medical Specialties",
            xaxis_title="Medical Specialty",
            yaxis_title="Probability (%)",
            yaxis=dict(
                range=[0, 100]
            ),
            template="plotly_dark",
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        st.subheader("🧠 Prediction Interpretation")

        st.info(
            f"The model classified this transcription as "
            f"**{prediction}** with a confidence of "
            f"**{confidence * 100:.2f}%**."
        )

        st.caption(
            "This model is intended for educational and "
            "research purposes and should not be used as "
            "a substitute for professional medical judgment."
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    '<div class="footer">'
    'Medical Specialty Classification • '
    'Bidirectional GRU • NLP Deep Learning Project'
    '</div>',
    unsafe_allow_html=True
)

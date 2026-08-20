# Medical Specialty Classification using BiGRU

A deep learning application that classifies medical transcription text into the most likely medical specialty using a **Bidirectional GRU (BiGRU)** neural network.

The project includes a trained NLP model and a **Streamlit web application** where users can enter medical transcription text and receive a predicted medical specialty.

## Overview

Medical transcriptions contain valuable information about a patient's medical visit, diagnosis, procedures, and treatment. Automatically identifying the medical specialty associated with a transcription can help organize and process large collections of clinical text.

In this project, I built an NLP classification model that learns patterns from medical transcription text and predicts one of **five medical specialty categories**.

The model uses an **Embedding layer followed by a Bidirectional GRU**, making it capable of understanding information from both directions of a text sequence.

## How It Works

The application follows a simple pipeline:

```text
Medical Transcription
        ↓
Text Preprocessing
        ↓
Tokenization
        ↓
Sequence Padding
        ↓
Embedding Layer
        ↓
Bidirectional GRU
        ↓
Dense Layers
        ↓
Softmax Output
        ↓
Predicted Medical Specialty
```

The input text is converted into numerical sequences using the trained tokenizer. These sequences are padded to a fixed length of **150 tokens** before being passed to the trained model.

The final layer produces probabilities for five medical specialty classes, and the class with the highest probability is returned as the prediction.

## Model Architecture

The trained model uses the following architecture:

* **Input:** Sequence of 150 tokens
* **Embedding:** Vocabulary size of 8,000 with 64-dimensional embeddings
* **Bidirectional GRU:** 64 units in each direction
* **Dropout:** 0.5
* **Dense:** 32 neurons with ReLU activation
* **Dropout:** 0.3
* **Output:** 5 classes with Softmax activation

The Bidirectional GRU allows the model to learn contextual information from both the beginning and the end of a transcription.

## Streamlit Application

The project includes an interactive Streamlit interface.

Users can:

1. Enter or paste medical transcription text.
2. Submit the text for prediction.
3. View the predicted medical specialty.
4. See the model's confidence/probability for the prediction.

The application loads the trained model and supporting artifacts from the `artifacts` directory.

## Project Structure

```text
medical-specialty-classification-bigru/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── artifacts/
    ├── medical_specialty_bigru.keras
    ├── tokenizer.pkl
    ├── label_encoder.pkl
    └── model_config.pkl
```

### Important Files

**`app.py`**
The Streamlit application that handles the user interface, text preprocessing, model loading, and prediction.

**`medical_specialty_bigru.keras`**
The trained Bidirectional GRU classification model.

**`tokenizer.pkl`**
The tokenizer used during training to convert text into numerical sequences.

**`label_encoder.pkl`**
The label encoder used to convert the model's numerical predictions back into medical specialty names.

**`model_config.pkl`**
Contains configuration information used by the application.

**`requirements.txt`**
Lists the Python packages required to run the project.

## Technologies Used

* Python
* TensorFlow
* Keras
* Natural Language Processing (NLP)
* Bidirectional GRU
* Streamlit
* NumPy
* Pandas
* Scikit-learn
* Plotly

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/sarika-845/medical-specialty-classification-bigru.git
cd medical-specialty-classification-bigru
```

### 2. Create a virtual environment

```bash
python -m venv myenv
```

Activate it on Windows:

```powershell
myenv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit application

Use:

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## Model Input

The model expects medical transcription text as input.

For example:

```text
Patient presents with chest pain and shortness of breath.
The patient has a history of coronary artery disease.
```

The application processes the text and returns the predicted medical specialty.

## Why BiGRU?

GRUs are well suited for sequence-based problems such as text classification because they can capture relationships between words across a sequence.

A **Bidirectional GRU** processes the sequence in both forward and backward directions. This gives the model additional context when interpreting medical terminology and phrases.

Compared with a basic RNN, GRUs can also handle longer dependencies more effectively while using fewer parameters than many traditional LSTM architectures.

## Reproducibility

The repository includes the trained model and preprocessing artifacts required for inference.

The TensorFlow and Keras versions are pinned in `requirements.txt` to reduce compatibility issues when loading the saved `.keras` model.

For best results, use the versions specified in `requirements.txt`.

## Limitations

This project is intended as an **NLP and machine learning demonstration**.

The predictions should not be considered medical advice, diagnosis, or a replacement for a qualified healthcare professional.

The model's performance also depends on the quality and distribution of the data used during training. It may not generalize perfectly to medical transcriptions from different sources or specialties.

## Future Improvements

Some possible improvements include:

* Training with a larger and more diverse medical dataset
* Supporting more medical specialty categories
* Improving text preprocessing
* Comparing BiGRU with LSTM and Transformer-based models
* Adding model evaluation metrics and confusion matrices
* Improving prediction explanations
* Deploying the application as a public web service
* Adding batch prediction for multiple transcriptions




import streamlit as st
import torch
import easyocr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from PIL import Image
import numpy as np
import time
from pdf2image import convert_from_bytes
import pandas as pd

# Page config

st.set_page_config(page_title="OCR Summarization System", layout="wide")
st.title("OCR-Based Abstractive Text Summarization")
st.write("Upload an image or PDF containing text. The system extracts text and generates a concise summary.")

# Load Models

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=torch.cuda.is_available())

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("./t5_finetuned_model")
    model = T5ForConditionalGeneration.from_pretrained("./t5_finetuned_model")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model

reader = load_ocr()
tokenizer, model = load_model()

# OCR Function

def extract_text_from_image(image):
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)

def extract_text_from_pdf(file):
    images = convert_from_bytes(file.read())
    full_text = ""
    for img in images:
        full_text += extract_text_from_image(img) + " "
    return full_text.strip()

# Summarization Function

def summarize_text(text):
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=120,
            min_length=40,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# File Upload Section

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["jpg", "png", "jpeg", "pdf"]
)

if uploaded_file:
    start_time = time.time()

    if uploaded_file.type == "application/pdf":
        st.info("PDF detected. Converting pages to images...")
        extracted_text = extract_text_from_pdf(uploaded_file)
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text_from_image(image)

    if not extracted_text.strip():
        st.error("No text detected.")
        st.stop()

    st.subheader("Extracted Text")
    st.write(extracted_text)

    with st.spinner("Generating Summary..."):
        summary = summarize_text(extracted_text)

    st.subheader("Generated Summary")
    st.success(summary)

    end_time = time.time()
    st.info(f"Processing Time: {round(end_time - start_time, 2)} seconds")

# Hidden Comparison Section (lstm vs t5)

with st.expander("➕ LSTM vs T5 Model Comparison"):

    # Example comparison (replace with your real values)
    lstm_results = {
        "rouge1": 0.02,
        "rouge2": 0.00,
        "rougeL": 0.01
    }

    t5_results = {
        "rouge1": 0.38,
        "rouge2": 0.16,
        "rougeL": 0.34
    }

    comparison = pd.DataFrame({
        "Model": ["LSTM (Vanilla)", "Fine-Tuned T5"],
        "ROUGE-1": [lstm_results["rouge1"], t5_results["rouge1"]],
        "ROUGE-2": [lstm_results["rouge2"], t5_results["rouge2"]],
        "ROUGE-L": [lstm_results["rougeL"], t5_results["rougeL"]],
    })

    st.dataframe(comparison)

    st.markdown("### 📊 Analysis")
    st.write("""
    - The LSTM model struggles due to lack of attention mechanism.
    - T5 benefits from self-attention and large-scale pretraining.
    - Transformers handle long documents significantly better.
    """)
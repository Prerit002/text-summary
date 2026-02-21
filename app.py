import streamlit as st
import torch
import fitz # PyMuPDF
import easyocr
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pdf2image import convert_from_bytes
from PIL import Image
import numpy as np
import time

# ============================================
# Configuration
# ============================================

MODEL_NAME = "Aryan-8878/text-summary"

st.set_page_config(
    page_title="Smart OCR Summarization",
    layout="wide"
)

st.title("📄 Smart OCR-Based Abstractive Text Summarization")
st.write("Upload an Image or PDF. The system extracts text and generates a proportional summary.")

# ============================================
# Load Models (CPU for Streamlit Cloud)
# ============================================

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'], gpu=False)

@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to("cpu")
    return tokenizer, model

reader = load_ocr()
tokenizer, model = load_model()

# ============================================
# OCR Functions
# ============================================

def extract_text_from_image(image):
    result = reader.readtext(np.array(image), detail=0)
    return " ".join(result)

def extract_text_from_pdf(file_bytes):
    text = ""

    # Try direct text extraction
    with fitz.open(stream=file_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()

    if len(text.strip()) > 50:
        return text.strip(), "Direct PDF Text Extraction"

    # Fallback to OCR
    images = convert_from_bytes(file_bytes)
    for img in images:
        text += extract_text_from_image(img)

    return text.strip(), "PDF Converted to Images + OCR"

# ============================================
# Long Text Summarization
# ============================================

def summarize_long_text(text, summary_ratio=0.15, chunk_size=1500):

    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    summaries = []

    for chunk in chunks:

        input_text = "summarize: " + chunk

        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            max_length=512,
            truncation=True
        )

        chunk_word_count = len(chunk.split())

        max_len = int(chunk_word_count * summary_ratio)
        max_len = max(80, min(max_len, 220))
        min_len = int(max_len * 0.6)

        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_len,
            min_length=min_len,
            num_beams=3,
            early_stopping=True
        )

        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        summaries.append(summary)

    combined_summary = " ".join(summaries)

    input_text = "summarize: " + combined_summary

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    )
    final_max = min(300, int(len(words) * summary_ratio))
    final_min = int(final_max * 0.6)

    final_ids = model.generate(
        inputs["input_ids"],
        max_length = final_max,
        min_length = final_min,
        num_beams=3,
        early_stopping = True
    )

    return tokenizer.decode(final_ids[0], skip_special_tokens=True)
    
# ============================================
# File Upload UI
# ============================================

uploaded_file = st.file_uploader(
    "Upload Image or PDF",
    type=["jpg", "jpeg", "png", "pdf"]
)

if uploaded_file:

    summary_ratio = st.slider(
        "📏 Summary Length Ratio",
        min_value=0.1,
        max_value=0.5,
        value=0.15,
        step=0.05,
        help="Lower = shorter summary, Higher = more detailed summary"
    )

    start_time = time.time()

    file_bytes = uploaded_file.read()

    if uploaded_file.type == "application/pdf":

        with st.spinner("Processing PDF..."):
            extracted_text, method = extract_text_from_pdf(file_bytes)

        st.subheader(f"📄 Extraction Method: {method}")
        st.write(extracted_text[:1000] + "...")

    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Extracting text from image..."):
            extracted_text = extract_text_from_image(image)

        st.subheader("📄 Extracted Text")
        st.write(extracted_text[:1000] + "...")

    if extracted_text.strip() == "":
        st.error("No text detected.")
    else:
        with st.spinner("Generating summary..."):
            summary = summarize_long_text(extracted_text, summary_ratio=summary_ratio)

        st.subheader("📝 Generated Summary")
        st.success(summary)

        # Metrics
        summary_word_count = len(summary.split())
        original_word_count = len(extracted_text.split())

        compression_ratio = summary_word_count / original_word_count

        st.markdown("---")
        col1, col2, col3 = st.columns(3)

        col1.metric("Original Words", original_word_count)
        col2.metric("Summary Words", summary_word_count)
        col3.metric("Compression", f"{((original_word_count)-(summary_word_count)/(original_word_count))*100}%")

    end_time = time.time()
    st.info(f"⏱ Processing Time: {round(end_time - start_time, 2)} seconds")

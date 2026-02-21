import streamlit as st
import torch
import fitz # PyMuPDF
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(page_title="PDF Summarization System", layout="wide")

st.title("📄 PDF Abstractive Text Summarization")
st.write("Upload a PDF document. The system extracts text and generates a concise summary using a fine-tuned Transformer model.")

# ---------------------------------
# Load Model (Cached)
# ---------------------------------
@st.cache_resource
def load_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-small")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-small")
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ---------------------------------
# PDF Text Extraction
# ---------------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# ---------------------------------
# Summarization Function
# ---------------------------------
def summarize_text(text):
    input_text = "summarize: " + text[:4000] # limit size for safety

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(model.device)

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=150,
        min_length=50,
        num_beams=4,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ---------------------------------
# Upload Section
# ---------------------------------
uploaded_file = st.file_uploader("📤 Upload PDF", type=["pdf"])

if uploaded_file:

    if st.button("Generate Summary"):

        start_time = time.time()

        with st.spinner("Extracting text from PDF..."):
            extracted_text = extract_text_from_pdf(uploaded_file)

        st.subheader("📄 Extracted Text Preview")
        st.write(extracted_text[:1000] + "...")

        with st.spinner("Generating summary..."):
            summary = summarize_text(extracted_text)

        st.subheader("📝 Generated Summary")
        st.success(summary)

        end_time = time.time()
        st.info(f"⏱ Processing Time: {round(end_time - start_time, 2)} seconds")

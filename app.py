import streamlit as st
from pipeline import pdf_to_images, extract_fields

st.set_page_config(layout="wide")

st.title("AI Signature Compliance Validator")

uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("Processing document...")

    pages = pdf_to_images("temp.pdf")

    all_fields = []

    for page_number, image in pages:
        result = extract_fields(image, page_number)
        all_fields.extend(result["fields"])

    st.subheader("Summary")
    st.write(f"Total Required Signatures: {len(all_fields)}")

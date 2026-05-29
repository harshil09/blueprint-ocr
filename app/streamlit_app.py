import requests
import streamlit as st

st.title("Engineering Figure Extraction")

uploaded_file = st.file_uploader(
    "Upload file",
    type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"],
)

# Placeholder for status
status_placeholder = st.empty()

if st.button("Process File"):

    if uploaded_file:

        # 1. show processing state
        status_placeholder.info("Processing... please wait ⏳")

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file,
                uploaded_file.type,
            )
        }

        try:
            # 2. call API
            response = requests.post(
                "http://127.0.0.1:8000/extract",
                files=files,
            )

            # 3. success handling
            if response.status_code == 200:
                status_placeholder.success("Processing completed successfully 🎉")
                st.json(response.json())
            else:
                status_placeholder.error(f"Error: {response.text}")

        except Exception as e:
            status_placeholder.error(f"Request failed: {str(e)}")

    else:
        st.warning("Please upload a file first")
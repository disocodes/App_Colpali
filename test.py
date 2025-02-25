import streamlit as st
import base64
from huggingface_hub import notebook_login
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, MllamaForConditionalGeneration
from PIL import Image
from io import BytesIO
import torch
import re
import os
import groq
from dotenv import load_dotenv

# Wrap the entire app in a function
def main():
    load_dotenv()
    upload_dir = "./doc"

    # Set page layout to wide
    st.set_page_config(layout="wide")

    st.title("Colpali Based Multimodal RAG App")

    # Create sidebar for configuration options
    with st.sidebar:
        st.header("Configuration Options")
        
        # Dropdown for selecting Colpali model
        colpali_model = st.selectbox(
            "Select Colpali Model",
            options=["vidore/colpali", "vidore/colpali-v1.2"]
        )
        
        # Dropdown for selecting Multi-Model LLM
        multi_model_llm = st.selectbox(
            "Select Multi-Model LLM",
            options=["llama2-70b-4096", "llama2-70b-chat", "mixtral-8x7b-32768"]
        )
        
        # Groq API Key input
        groq_api_key = st.text_input("Enter Groq API Key", type="password")
        
        # File upload button
        uploaded_file = st.file_uploader("Choose a Document", type=["pdf"])

    # Initialize Groq client
    if groq_api_key:
        client = groq.Client(api_key=groq_api_key)
    else:
        st.warning("Please enter your Groq API key in the sidebar.")

    # Main content layout
    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("### Uploaded Document")
            save_path = os.path.join(upload_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File saved: {uploaded_file.name}")
            
            @st.cache_resource
            def load_models(colpali_model):
                RAG = RAGMultiModalModel.from_pretrained(colpali_model, verbose=10)
                return RAG
            
            RAG = load_models(colpali_model)
            
            @st.cache_data
            def create_rag_index(image_path):
                RAG.index(
                    input_path=image_path,
                    index_name="image_index",
                    store_collection_with_index=True,
                    overwrite=True,
                )
            
            create_rag_index(save_path)
            
        with col2:
            # Text input for the user query
            text_query = st.text_input("Enter your text query")

            # Search and Extract Text button
            if st.button("Search and Extract Text"):
                if text_query and groq_api_key:
                    results = RAG.search(text_query, k=1, return_base64_results=True)

                    image_data = base64.b64decode(results[0].base64)
                    image = Image.open(BytesIO(image_data))
                    st.image(image, caption="Result Image", use_column_width=True)

                    # Prepare the message for Groq's Llama model
                    messages = [
                        {"role": "system", "content": "You are a helpful assistant analyzing images and text."},
                        {"role": "user", "content": f"Analyze this image and answer the following question: {text_query}"}
                    ]

                    try:
                        response = client.chat.completions.create(
                            model=multi_model_llm,
                            messages=messages,
                            max_tokens=300,
                            temperature=0.7
                        )
                        output = response.choices[0].message.content
                        
                        st.subheader("Query with LLM Model")
                        st.markdown(output, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
                elif not groq_api_key:
                    st.warning("Please enter your Groq API key in the sidebar.")
                else:
                    st.warning("Please enter a query.")
    else:
        st.info("Upload a document to get started.")

if __name__ == "__main__":
    main()

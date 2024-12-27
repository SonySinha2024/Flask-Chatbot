from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import UnstructuredURLLoader
import logging as log
import httpx
from concurrent.futures import ThreadPoolExecutor

# Configure logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = log.getLogger(__name__)

url ="https://www.testrigtechnologies.com/"
# Load environment variables
load_dotenv()
google_api_key = os.getenv('google_api_key')
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-pro")

google_api_key = "AIzaSyDfVIOcSaPeaRPbAA_pMbHJe9YZAySy-Ig"

app = Flask(__name__)

# System prompt for Gemini
system_prompt = """
You are a RAG-based chatbot and must answer all questions using the content of the provided URL.
The URL is "https://www.testrigtechnologies.com/". You must respond with details such as office locations, services, contact information, or anything available on this URL.
If the question is unrelated, respond with:
"I am sorry, please ask queries only related to the context of the URL or the software testing and development domain."
"""
def fetch_text_from_url(url):
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
            if response.status_code != 200:
                log.error(f"Failed to fetch URL. Status code: {response.status_code}")
                return ""
            soup = BeautifulSoup(response.content, "lxml")
            text = " ".join(p.get_text() for p in soup.find_all("p"))
            log.info(f"Fetched text: {text[:500]}")  # Log the first 500 characters of the text
            if not text.strip():
                log.error("No meaningful text found on the page.")
                return ""
            return text
    except Exception as e:
        log.error(f"Error fetching URL: {e}")
        return ""

# Fetch and process content from a URL
# def fetch_text_from_url(url):
#     try:
#         with httpx.Client(timeout=10) as client:
#             response = client.get(url)
#             soup = BeautifulSoup(response.content, "lxml")

#             # Extract unstructured data
#             paragraphs = " ".join([p.get_text() for p in soup.find_all("p")])

#             # Extract structured data (e.g., addresses, emails)
#             contact_info = " ".join([tag.get_text() for tag in soup.find_all(["address", "a"]) if "@" in tag.text or tag.text.strip()])
#             return f"{paragraphs}\n{contact_info}"
#     except Exception as e:
#         log.error(f"Error fetching URL: {e}")
#         return ""

# def initialize_vectordb(url):
#     text = fetch_text_from_url(url)
#     text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
#     chunks = text_splitter.split_text(text)
#     if not chunks:
#       log.error("Text chunks are empty. Check the URL content or text splitter configuration.")
#       raise ValueError("No text data available for embedding.")

#     embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
#     test_text = "This is a test sentence."
#     embedding = embeddings.embed_query(test_text)
#     if not embedding:
#        log.error("Failed to generate embeddings. Check the embedding model configuration.")
#        raise ValueError("Embedding generation failed.")
#     vectordb = Chroma.from_texts(chunks, embedding=embeddings, persist_directory="db")
#     vectordb.persist()
#     return vectordb

# Initialize Vector Database
vectordb = initialize_vectordb("https://www.testrigtechnologies.com/")

# Generate response with RAG approach
def get_gemini_response(question):
    try:
        # Search for relevant context using embeddings
        results = vectordb.similarity_search(question, k=3)
        context = " ".join([result.page_content for result in results])

        chat = model.start_chat(history=[])
        response = chat.send_message(f"{system_prompt}\nContext: {context}\nQuestion: {question}", stream=True)
        return "".join([chunk.text for chunk in response])
    except Exception as e:
        log.error(f"Error generating response: {e}")
        return "Sorry, I couldn't fetch a response. Please try again later."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message']
    response = get_gemini_response(user_input)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)


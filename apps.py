from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
import logging as log
import json
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
import httpx
from concurrent.futures import ThreadPoolExecutor

# Configure logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = log.getLogger(__name__)

# Load environment variables
load_dotenv()
google_api_key = os.getenv('google_api_key')

google_api_key = "AIzaSyDfVIOcSaPeaRPbAA_pMbHJe9YZAySy-Ig"
# Configure Gemini API
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-pro")

# Initialize Flask app
app = Flask(__name__)

# Save chat history and system prompt
chat_history_file = "chat_history.json"
system_prompt = """
You are a knowledgeable chatbot that responds only to questions related to software testing and development or based on the content of an uploaded URL. 
If the query is unrelated to software testing, development, or the URL content, respond with:
"I am sorry, please query only related to the context of the URL or based on software testing and development domain."
Answer the following query based solely on the URL context if available, or software testing and development domain as a fallback.
"""

def save_chat_history(chat_history):
    try:
        with open(chat_history_file, "w") as f:
            json.dump(chat_history, f, indent=4)
    except Exception as e:
        log.error(f"Failed to save chat history: {e}")

def load_chat_history():
    if os.path.exists(chat_history_file):
        try:
            with open(chat_history_file, "r") as f:
                return json.load(f)
        except Exception as e:
            log.error(f"Failed to load chat history: {e}")
    return []

chat_history = load_chat_history()
 Left column with content 

def fetch_text_from_url(url):
    try:
        with httpx.Client(timeout=10) as client:
            response = client.get(url)
            soup = BeautifulSoup(response.content, "lxml")
            text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text
    except Exception as e:
        log.error(f"Error fetching URL: {e}")
        return ""


def process_url_content(url):
    try:
        with ThreadPoolExecutor() as executor:
            future = executor.submit(fetch_text_from_url, url)
            text = future.result()
        text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)
        return chunks
    except Exception as e:
        log.error(f"Error processing URL content: {e}")
        return []


def initialize_vectordb(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb


def get_gemini_response(question, context=None):
    try:
        chat_session = model.start_chat(history=chat_history)
        if context:
            question = f"{context}: {question}"
        response_stream = chat_session.send_message(question, stream=True)
        response_text = "".join(chunk.text for chunk in response_stream)
        return response_text
    except Exception as e:
        log.error(f"Error getting response from Gemini: {e}")
        return "Sorry, I couldn't fetch a response. Please try again later."

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message']
    context_url = request.form.get('context_url')

    context = None
    if context_url:
        context = fetch_text_from_url(context_url)
        if not context:
            return jsonify({'response': "Failed to fetch content from the provided URL."})

 
    response = get_gemini_response(user_input, context)

    
    chat_history.append({"user": user_input, "response": response})
    save_chat_history(chat_history)

    return jsonify({'response': response})

@app.route('/history', methods=['GET'])
def get_history():
    return jsonify(chat_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    global chat_history
    chat_history = []
    save_chat_history(chat_history)
    return jsonify({'message': "Chat history cleared successfully."})

@app.route('/download_history', methods=['GET'])
def download_history():
    try:
        return jsonify(chat_history), 200, {
            'Content-Disposition': 'attachment;filename=chat_history.json',
            'Content-Type': 'application/json'
        }
    except Exception as e:
        log.error(f"Failed to download chat history: {e}")
        return jsonify({'message': "Error downloading chat history."}), 500

if __name__ == "__main__":
    app.run(debug=True)

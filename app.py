from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
import re
import logging as log
import json
import httpx
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import requests

# Configure logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = log.getLogger(__name__)

# Load environment variables
load_dotenv()
google_api_key = os.getenv('google_api_key')

# if not google_api_key:
#     log.error("Google Gemini API key is missing.")
#     raise EnvironmentError("Google Gemini API key is required in the .env file.")

google_api_key = "AIzaSyDfVIOcSaPeaRPbAA_pMbHJe9YZAySy-Ig"
# Configure Gemini API
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-pro")

#genai.configure(api_key=google_api_key)
#model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat()

## Preprocess query
def preprocess_query(user_query):
    corrected_query = user_query.strip().lower()
    if "software testing" in corrected_query:
        rephrased_query = f"Could you provide information about {corrected_query.replace('software testing', 'software testing related topics')}?"
    else:
        rephrased_query = f"Can you tell me more about {corrected_query}?"
    return rephrased_query

# Initialize Flask app
app = Flask(__name__)
# Save chat history and system prompt
#chat_history_file = "chat_history.json"
system_prompt = """
You are a Rag based chatbot that responds only to questions related to software testing and development or based on the content of an uploaded URL. 
You will answer everything based on the content of the URL link of any web.
url link is "https://www.testrigtechnologies.com/" and you will answer everything related to this url.
If the query is unrelated to software testing, development or the URL content then respond with:
"I am sorry, please ask query only related to the context of the URL or the software testing and development domain."
Answer the following query based solely on the URL content if available, or based on software testing and development domain as a fallback.
You can answer sensitive information such as email id, contact no, address, ceo name, location of various Testrig offices.
"""

url="https://www.testrigtechnologies.com/"

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
    
## Initialize vector database and embeddings
def initialize_vectordb(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb

def get_gemini_response(question, context=None, prompt_type="qna"):
    chat = model.start_chat(history=[])
    if context:
        question = f"{context}: {question}"
    response = chat.send_message(question, stream=True)
    response_text = "".join([chunk.text for chunk in response])
    response_text = re.sub(r'\*\*', '', response_text)
    response_text = re.sub(r'\* (.*?)\*\*:', r'\1:', response_text)  
    response_text = re.sub(r'(\w+:)', r'\n\1', response_text)  

    # Ensure proper sentence formatting
    formatted_response = (
        response_text.replace(". ", ".\n"),
        response_text.replace("* ", " ")
        .replace(" I am", "\nI am")
        .strip()
    )
    return formatted_response

# def get_gemini_response(question, context=None, prompt_type="qna"):
#     chat = model.start_chat(history=[])
#     if context:
#         question = f"{context}: {question}"
#     response = chat.send_message(question, stream=True)
#     response_text = "".join([chunk.text for chunk in response])

#     # Format response: Add a newline after each sentence and handle bullets
#     formatted_response = (
#         response_text.replace(". ", ".\n")
#         .replace("* ", "\n* ")
#         .replace(" I am", "\nI am")
#     )
#     return formatted_response

# def get_gemini_response(question, context=None, prompt_type="qna"):
#     chat = model.start_chat(history=[])
#     if context:
#         question = f"{context}: {question}"
#     response = chat.send_message(question, stream=True)
#     response_text = "".join([chunk.text for chunk in response])

#     # Format response: Add a new line after each sentence
#     formatted_response = "\n".join(sentence.strip() for sentence in response_text.split('. ') if sentence.strip())
#     return formatted_response

# Gemini response generator
# def get_gemini_response(question, context=None, prompt_type="qna"):
#     chat = model.start_chat(history=[])
#     if context:
#         question = f"{context}: {question}"
#     response = chat.send_message(question, stream=True)
#     response_text = "".join([chunk.text for chunk in response])
#     return response_text

# def get_gemini_response(question, context=None):
#     try:
#         chat_session = model.start_chat(history=chat_history)
#         if context:
#             question = f"{context}: {question}"
#         response_stream = chat_session.send_message(question, stream=True)
#         response_text = "".join(chunk.text for chunk in response_stream)
#         return response_text
#     except Exception as e:
#         log.error(f"Error getting response from Gemini: {e}")
#         return "Sorry, I couldn't fetch a response. Please try again later."

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
  
    # chat_history.append({"user": user_input, "response": response})
    # save_chat_history(chat_history)

    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)

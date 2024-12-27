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
from langchain_community.document_loaders import UnstructuredURLLoader
#from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
#from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain.embeddings import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForCausalLM
from concurrent.futures import ThreadPoolExecutor
import requests
from langchain_community.vectorstores import Chroma

## Configure logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = log.getLogger(__name__)

## Load environment variables from .env
config_file_path = 'C:/frontend/Flask-Chatbot/config/.env'
load_dotenv(dotenv_path=config_file_path)

google_api_key = "AIzaSyDfVIOcSaPeaRPbAA_pMbHJe9YZAySy-Ig"

## google_api_key = os.getenv('google_api_key')
#google_api_key = ""
if not google_api_key:
    log.error("Required API keys are missing from environment variables.")
    raise EnvironmentError("Required API keys are missing from environment variables.")

genai.configure(api_key=google_api_key)
model = genai.GenerativeModel("gemini-pro")
chat = model.start_chat()

app = Flask(__name__)

# Preprocess query
def preprocess_query(user_query):
    corrected_query = user_query.strip().lower()
    if "software testing" in corrected_query:
        rephrased_query = f"Could you provide information about {corrected_query.replace('software testing', 'software testing related topics')}?"
    else:
        rephrased_query = f"Can you tell me more about {corrected_query}?"
    return rephrased_query

# Fetch and process content from URL
def fetch_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    text = " ".join(p.get_text() for p in soup.find_all("p"))
    return text

# Initialize vector database and embeddings
def initialize_vectordb(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    text_splitter = CharacterTextSplitter(separator='\n', chunk_size=1000, chunk_overlap=200)
    text_chunks = text_splitter.split_documents(data)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(text_chunks, embedding=embeddings, persist_directory="db")
    vectordb.persist()
    return vectordb

# def fetch_text_from_url(url):
#     try:
#         with httpx.Client(timeout=10) as client:
#             response = client.get(url)
#             soup = BeautifulSoup(response.content, "lxml")
#             text = " ".join(p.get_text() for p in soup.find_all("p"))
#         return text
#     except Exception as e:
#         log.error(f"Error fetching URL: {e}")
#         return ""

def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "lxml")  # Test lxml parser
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text
    except Exception as e:
        print(f"Error: {e}")
        return ""

url = "https://www.testrigtechnologies.com/"
content = fetch_text_from_url(url)
print(content[:500])  

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
    
# Gemini response generator
def get_gemini_response(question, context=None, prompt_type="qna"):
    chat = model.start_chat(history=[])
    if context:
        question = f"{context}: {question}"
    response = chat.send_message(question, stream=True)
    response_text = "".join([chunk.text for chunk in response])
    return response_text.replace("**" , "\n")

url_input ="https://www.testrigtechnologies.com/"
if url_input:
    try:
        context_text = fetch_text_from_url(url_input)
        vectordb = initialize_vectordb([url_input])
        retriever = vectordb.as_retriever()
        #st.session_state['context'] = context_text
        print("URL content loaded successfully.")
    except Exception as e:
        log.error(f"Failed to fetch content from URL: {e}")

@app.route('/')
def home():
    return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.form['message']
#     response = get_gemini_response(user_input)
#     #return str(response)
#     return jsonify({'response': response})


#@app.route('/ask', methods=['POST'])
# def ask():
#     import re
#     user_input = request.form['message']
#     response = get_gemini_response(user_input)  # Get the chatbot's response

#     # Clean and format the response
#     raw_text = response.get('response', '')  # Extract the text part of the JSON
#     cleaned_text = re.sub(r'\*', '', raw_text)  # Remove the '*' characters
#     formatted_text = cleaned_text.replace('. ', '.\n')  # Add newlines after sentences
    
#     return jsonify({'response': formatted_text})  # Return the formatted response

#stringified_response = json.dumps(response)  # Convert JSON to a string
#return stringified_response  # Return the stringified JSON


# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.form['message']
#     response = get_gemini_response(user_input)  # This returns a JSON response
#     stringified_response = json.dumps(response)  # Convert JSON to a string
#     return stringified_response  # Return the stringified JSON

@app.route('/ask', methods=['POST'])
def ask():
    import re
    user_input = request.form['message']
    response = get_gemini_response(user_input)  # Get the chatbot's response

    # Clean and format the response
    cleaned_text = re.sub(r'\*', '', response)  # Remove the '*' characters
    formatted_text = cleaned_text.replace('. ', '.\n')  # Add newlines after sentences
    
    return jsonify({'response': formatted_text})  # Return the formatted response

if __name__ == "__main__":
    app.run(debug=True)

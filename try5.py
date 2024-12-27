from flask import Flask, request, jsonify, render_template
import os
from dotenv import load_dotenv
import google.generativeai as genai
from bs4 import BeautifulSoup
import requests
import re
import logging as log
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from concurrent.futures import ThreadPoolExecutor

# Configure logging
log.basicConfig(level=log.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
log = log.getLogger(__name__)

# Load environment variables
config_file_path = 'C:/frontend/Flask-Chatbot/config/.env'
load_dotenv(dotenv_path=config_file_path)

google_api_key = "AIzaSyDfVIOcSaPeaRPbAA_pMbHJe9YZAySy-Ig"
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
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")
        text = " ".join(p.get_text() for p in soup.find_all("p"))
        return text
    except Exception as e:
        log.error(f"Error fetching text from URL: {e}")
        return ""

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

def few_shot_qna(question):
    examples = [
        ("Do you offer automated testing services?", 
         "Yes, we offer Automated testing services. You can learn more about our services here: <a href='https://www.testrigtechnologies.com/automation-testing/' target='_blank'>Automated Testing Services</a>"),
        ("Do you offer training for in-house testing teams?", 
         "Yes, we have training sessions for our team."),
        ("How to contact Testrig Technologies?",
         """You can connect with us through the following channels:<br>
         1 Schedule a meeting directly with our team: <a href='https://www.testrigtechnologies.com/about-us/' target='_blank'>About Us</a><br>
         2 Email us at: <a href='mailto:info@testrigtechnologies.com'>info@testrigtechnologies.com</a><br>
         3 Submit your inquiry via our contact form: <a href='https://www.testrigtechnologies.com/contact-us/' target='_blank'>Contact Us</a>""")
    ]
    formatted_examples = "\n".join([f"Q: {ex[0]}\nA: {ex[1]}" for ex in examples])
    return f"{formatted_examples}\nQ: {question}"

# def few_shot_qna(question):
#     examples = [
#         ("Do you offer automated testing services?", 
#          "Yes, we offer Automated testing services. You can learn more about our services here: <a href='https://www.testrigtechnologies.com/automation-testing/' target='_blank'>Automated Testing Services</a>"),
#         ("Do you offer training for in-house testing teams?", 
#          "Yes, we have training sessions for our team."),
#         ("How to contact Testrig Technologies?",
#          "You can connect with us through the following channels: Schedule a meeting directly with our team <a href='https://www.testrigtechnologies.com/about-us/' target='_blank'>About Us</a> or Email us at: <a href='mailto:info@testrigtechnologies.com'>info@testrigtechnologies.com</a> or Submit your inquiry via our contact form: <a href='https://www.testrigtechnologies.com/contact-us/' target='_blank'>Contact Us</a>")
#     ]
#     formatted_examples = "\n".join([f"Q: {ex[0]}\nA: {ex[1]}" for ex in examples])
#     return f"{formatted_examples}\nQ: {question}"


# # Few-shot Q&A prompt
# def few_shot_qna(question):
#     examples = [
#         ("Do you offer automated testing services?", 
#          "Yes, we offer Automated testing services. Please find more about our services at https://www.testrigtechnologies.com/automation-testing/"),
#         ("Do you offer training for in-house testing teams?", 
#          "Yes, we have training sessions for our team."),
#         ("How to contact Testrig Technologies?",
#          "You can connect with us through the following channels: Schedule a meeting directly with our team https://www.testrigtechnologies.com/about-us/ or Email us at: info@testrigtechnologies.com or Submit your inquiry via our contact form: https://www.testrigtechnologies.com/contact-us/")
#     ]
#     formatted_examples = "\n".join([f"Q: {ex[0]}\nA: {ex[1]}" for ex in examples])
#     return f"{formatted_examples}\nQ: {question}"

# Summarization template
def summarization_template(text):
    return f"Summarize the following information based on information found in the URL https://www.testrigtechnologies.com/:\n\n{text}"

# Q&A template
def qna_template(question):
    return f"Answer the following Testrig Technology-related question in simple terms:\n\n{question}"

# Gemini response generator
def get_gemini_response(question, context=None, prompt_type="qna"):
    chat = model.start_chat(history=[])
    if context:
        question = f"{context}: {question}"
    if prompt_type == "summarization":
        question = summarization_template(question)
    elif prompt_type == "few_shot_qna":
        question = few_shot_qna(question)
    else:
        question = qna_template(question)
    response = chat.send_message(question, stream=True)
    response_text = "".join([chunk.text for chunk in response])
    return response_text.replace("**", "\n")

# Process URL content
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

# Initialize context
url_input = "https://www.testrigtechnologies.com/"
if url_input:
    try:
        context_text = fetch_text_from_url(url_input)
        vectordb = initialize_vectordb([url_input])
        retriever = vectordb.as_retriever()
        log.info("URL content loaded successfully.")
    except Exception as e:
        log.error(f"Failed to fetch content from URL: {e}")

@app.route('/')
def home():
    return render_template('index.html', welcome_message="Testrig Technologies is QA and Software Testing Company and we have all solutions for all QA and Testing needs")

# @app.route('/ask', methods=['POST'])
# def ask():
#     user_input = request.json['message']
#     response = get_gemini_response(user_input, prompt_type="few_shot_qna")
#     return jsonify({'response': response})

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['message']
    response = get_gemini_response(user_input, prompt_type="few_shot_qna")

    # Clean and format the response
    cleaned_text = re.sub(r'\*', '', response)  # Remove the '*' characters
    formatted_text = cleaned_text.replace('. ', '.\n')  # Add newlines after sentences

    return jsonify({'response': formatted_text})

if __name__ == "__main__":
    app.run(debug=True)


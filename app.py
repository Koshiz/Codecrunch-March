from flask import Flask, request, render_template, redirect, url_for
import os
import numpy as np
import openai
from pinecone import Pinecone
from dotenv import load_dotenv
from utils.pdf_parser import extract_text_from_pdf
from utils.chunker import chunk_text
import re

# Load environment variables from .env file
load_dotenv()

# Retrieve environment variables
# AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
# AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
# AZURE_DEPLOYMENT_NAME = os.getenv("AZURE_DEPLOYMENT_NAME")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


AZURE_OPENAI_ENDPOINT= "https://koshi-m8fjen9l-eastus.cognitiveservices.azure.com/openai/deployments/gpt-4o-mini-codecrunch/chat/completions?api-version=2025-01-01-preview"
PINECONE_API_KEY ="pcsk_qxG6y_7mF9ZdL9ssuoDJ7GPKzNQuJcWLZW8972xbfobnjtvTfiQeghhDJVzMXxGNJAFSD" 
AZURE_OPENAI_KEY  = "8n2pZ5pvRR38rijG4zJZyqmSFBiQeaMmy3kYmk7dXvJRsDWszF36JQQJ99BCACYeBjFXJ3w3AAAAACOGKr9B"
PINECONE_INDEX_NAME = "codecrunch-march"
AZURE_DEPLOYMENT_NAME = "gpt-4o-mini-codecrunch"

app = Flask(__name__)

UPLOAD_FOLDER = 'Cvs/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Azure OpenAI Configuration
openai.api_type = "azure"
openai.api_base = AZURE_OPENAI_ENDPOINT
openai.api_version = "2023-05-15"
openai.api_key = AZURE_OPENAI_KEY

# Pinecone Configuration
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create Pinecone index if not exists
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,  # Adjust based on embedding model
#         metric='cosine'
#     )

#     f PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=1536,  # Based on embedding model
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud='aws',   # or 'azure' if you use Azure
#             region='us-west-2'  # Replace with your Pinecone region
#         )
#     )

index = pc.Index(PINECONE_INDEX_NAME)

# Generate Embedding using Azure OpenAI
def generate_embedding(text):
    response = openai.Embedding.create(
        input=text,
        engine=AZURE_DEPLOYMENT_NAME
    )
    return response['data'][0]['embedding']

# Extract email from text
def extract_email(text):
    match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    return match.group(0) if match else ''

# Extract phone number from text
def extract_phone(text):
    match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{2,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}', text)
    return match.group(0) if match else ''

# Extract candidate name (simple: first non-empty line)
def extract_name(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else ''

@app.route('/')
def index_page():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['resume']
    if file and file.filename.endswith('.pdf'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Process PDF
        text = extract_text_from_pdf(filepath)
        candidate_name = extract_name(text)
        candidate_email = extract_email(text)
        candidate_phone = extract_phone(text)

        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            embedding = generate_embedding(chunk)
            vector_id = f"{file.filename}_{idx}"
            metadata = {
                "file_name": file.filename,
                "chunk_index": idx,
                "chunk_text": chunk,
                "candidate_name": candidate_name,
                "candidate_email": candidate_email,
                "candidate_phone": candidate_phone
            }
            index.upsert([(vector_id, embedding, metadata)])

    return redirect(url_for('index_page'))

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    query_embedding = generate_embedding(query)
    search_results = index.query(vector=query_embedding, top_k=5, include_metadata=True)

    results = []
    for match in search_results['matches']:
        results.append(match['metadata'])

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

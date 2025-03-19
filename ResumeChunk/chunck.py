from pinecone import Pinecone  # Only import Pinecone
import openai
import os
from dotenv import load_dotenv
import fitz  # PyMuPDF for extracting text
import tiktoken

# Load environment variables from .env file (if applicable)
load_dotenv()

# Azure OpenAI credentials - FIXED CONFIGURATION
client = openai.AzureOpenAI(
    api_key="8n2pZ5pvRR38rijG4zJZyqmSFBiQeaMmy3kYmk7dXvJRsDWszF36JQQJ99BCACYeBjFXJ3w3AAAAACOGKr9B",
    api_version="2023-05-15",
    azure_endpoint="https://koshi-m8fjen9l-eastus.cognitiveservices.azure.com/"
)

# Print the OpenAI API key (for debugging purposes)
print("API Key configured ✓")

# Hardcoded Pinecone credentials
pc = Pinecone(api_key="pcsk_qxG6y_7mF9ZdL9ssuoDJ7GPKzNQuJcWLZW8972xbfobnjtvTfiQeghhDJVzMXxGNJAFSD")

# Create or connect to an existing index in Pinecone
index_name = "codecrunch-march"  # You can change this name
index = pc.Index(index_name)  # Correct usage: Access Index through the pc client

# Step 1: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text("text") for page in doc])
    doc.close()
    return text.strip()

# Step 2: Chunk the Text
def chunk_text(text, chunk_size=512, overlap=50):
    """Splits text into token-based chunks."""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))

    return chunks

# Step 3: Get Embeddings from Azure OpenAI - FIXED FUNCTION
def get_embeddings(text):
    """Sends text to Azure OpenAI embedding model and returns embeddings."""
    response = client.embeddings.create(
        model="text-embedding-ada-002-codecrunch",  # Your deployment name
        input=[text]  # Pass the input text as a list
    )
    # Correctly access embedding data in the response
    return response.data[0].embedding


# Step 4: Store Embeddings in Pinecone
def store_in_pinecone(chunks):
    """Stores the chunk embeddings in Pinecone."""
    batch_size = 100  # Process in batches to avoid memory issues
    total_chunks = len(chunks)
    
    for batch_start in range(0, total_chunks, batch_size):
        batch_end = min(batch_start + batch_size, total_chunks)
        current_batch = chunks[batch_start:batch_end]
        
        vectors_to_upsert = []
        
        for i, chunk in enumerate(current_batch):
            chunk_id = f"chunk_{batch_start + i}"
            try:
                embedding = get_embeddings(chunk)
                vectors_to_upsert.append((chunk_id, embedding, {"chunk_id": batch_start + i, "text": chunk}))
            except Exception as e:
                print(f"Error processing chunk {batch_start + i}: {str(e)}")
                continue
        
        if vectors_to_upsert:
            index.upsert(vectors=vectors_to_upsert)
            print(f"Stored batch {batch_start//batch_size + 1} with {len(vectors_to_upsert)} chunks in Pinecone ✅")

# Process the PDF and store the embeddings in Pinecone
pdf_path = "./Resumes/Careers - Movindu Liyanage.pdf"  # Path to your PDF file
text = extract_text_from_pdf(pdf_path)  # Extract text from PDF
chunks = chunk_text(text)  # Chunk the text
store_in_pinecone(chunks)  # Store embeddings in Pinecone

print("✅ PDF processed and stored in Pinecone!")
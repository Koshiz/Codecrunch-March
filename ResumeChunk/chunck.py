import fitz  # PyMuPDF for PDF processing
import tiktoken  # For token-based chunking
import textwrap

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    doc = fitz.open("./Resumes/Careers - Movindu Liyanage.pdf")
    text = ""

    for page in doc:
        text += page.get_text("text") + "\n"

    doc.close()
    return text.strip()

def chunk_text(text, chunk_size=512, overlap=50):
    """Chunks text into smaller parts based on token size."""
    encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI tokenizer
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size - overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(encoding.decode(chunk))

    return chunks

# Example usage
pdf_path = "cv.pdf"
text = extract_text_from_pdf(pdf_path)
chunks = chunk_text(text, chunk_size=512)

# Print the chunks
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{textwrap.fill(chunk, width=80)}\n")

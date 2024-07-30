# ChatWithDocs.py
import PyPDF2
import requests
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Function to parse PDFs
def parse_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to fetch text from URLs
def fetch_url(url):
    response = requests.get(url)
    return response.text

# Chat function for documents
def chat_with_documents(files, url, text_input, query):
    # Parse each file and extract text
    documents = []

    if files is not None:
        for file in files:
            if file.name.endswith('.pdf'):
                documents.append(parse_pdf(file))
            elif file.name.endswith('.txt'):
                documents.append(file.read().decode('utf-8'))

    if url:
        documents.append(fetch_url(url))

    if text_input:
        documents.append(text_input)

    # Create an embedding and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(documents, embeddings)

    # Create a RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        chain_type="stuff", retriever=vector_store.as_retriever()
    )

    # Get response
    response = qa_chain.run(query)
    return response

# Gradio interface for Chat with Documents
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Chat with Your Documents")
        with gr.Tab("Upload Documents"):
            with gr.Row():
                files = gr.File(label="Upload PDFs or Text Files", file_count="multiple")
                url = gr.Textbox(label="Enter URL")
                text_input = gr.Textbox(label="Enter Text")

            query = gr.Textbox(label="Enter your query")
            output = gr.Textbox(label="Response")

            btn = gr.Button("Submit")
            btn.click(chat_with_documents, inputs=[files, url, text_input, query], outputs=output)

    return interface

if __name__ == "__main__":
    create_interface().launch()





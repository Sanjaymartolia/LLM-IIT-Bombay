# InstiGPT.py
import PyPDF2
import gradio as gr
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory

# Load and parse the PDF dataset
def load_instidata():
    text = ""
    with open("data/Instidata.pdf", "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Chat function for InstiGPT
def instigpt_chat(query, history):
    # Load dataset
    instidata = load_instidata()

    # Create an embedding and vector store
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts([instidata], embeddings)

    # Create a RAG chain with memory
    memory = ConversationBufferMemory()
    qa_chain = ConversationalRetrievalChain(
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    # Get response
    response = qa_chain.run(query, history=history)
    return response

# Gradio interface for InstiGPT
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# InstiGPT Chatbot")
        with gr.Tab("Chat with InstiGPT"):
            query = gr.Textbox(label="Enter your query")
            history = gr.State([])
            output = gr.Textbox(label="Response")

            btn = gr.Button("Submit")
            btn.click(instigpt_chat, inputs=[query, history], outputs=output)

    return interface

if __name__ == "__main__":
    create_interface().launch()


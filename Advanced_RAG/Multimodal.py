# Multimodal.py
import fitz  # PyMuPDF
import gradio as gr
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Function to extract text and images from PDF
def extract_multimodal_data(file_path):
    doc = fitz.open(file_path)
    text = ""
    images = []

    for page in doc:
        text += page.get_text()
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append(base_image["image"])

    return text, images

# Multimodal chat function
def multimodal_chat(query):
    # Extract data from PDF
    text, images = extract_multimodal_data("data/Candlestick_Patterns_Multimodal_Data_SOC166.pdf")

    # Create an embedding and vector store for text
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts([text], embeddings)

    # Create a RAG chain for text
    qa_chain = RetrievalQA.from_chain_type(
        chain_type="stuff", retriever=vector_store.as_retriever()
    )

    # Get response from text
    text_response = qa_chain.run(query)

    # Return images (currently placeholder)
    image_response = images[:3]  # Return first three images

    return text_response, image_response

# Gradio interface for Multimodal RAG Chat
def create_interface():
    with gr.Blocks() as interface:
        gr.Markdown("# Multimodal RAG Chat")
        with gr.Tab("Chat with Text and Images"):
            query = gr.Textbox(label="Enter your query")
            text_output = gr.Textbox(label="Text Response")
            image_output = gr.Gallery(label="Image Response")

            btn = gr.Button("Submit")
            btn.click(multimodal_chat, inputs=[query], outputs=[text_output, image_output])

    return interface

if __name__ == "__main__":
    create_interface().launch()




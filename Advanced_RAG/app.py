# app.py
import gradio as gr
import ChatWithDocs
import InstiGPT
import Multimodal
import MultimodalChat

# Create the main Gradio interface
with gr.Blocks() as main_interface:
    gr.Markdown("# Advanced RAG Chatbot Web App")
    with gr.Tab("Chat with Your Documents"):
        ChatWithDocs.create_interface()

    with gr.Tab("InstiGPT"):
        InstiGPT.create_interface()

    with gr.Tab("Multimodal RAG Chat"):
        Multimodal.create_interface()

    with gr.Tab("Multimodal Chat"):
        MultimodalChat.create_interface()

if __name__ == "__main__":
    main_interface.launch()






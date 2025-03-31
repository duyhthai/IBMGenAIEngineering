from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import gradio as gr


## LLM
def get_llm():
    model_id = "mistralai/Mistral-7B-Instruct-v0.1"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map="auto"
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
    )

    return HuggingFacePipeline(pipeline=pipe)


## Document loader
def document_loader(file):
    loader = PyPDFLoader(file.name)
    loaded_document = loader.load()
    return loaded_document


## Text splitter
def text_splitter(data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(data)
    return chunks


## Vector db
def vector_database(chunks):
    embedding_model = huggingface_embedding()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


## Embedding model
def huggingface_embedding():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    huggingface_embedding = HuggingFaceEmbeddings(model_name=model_name)
    return huggingface_embedding


## Retriever
def retriever(file):
    splits = document_loader(file)
    chunks = text_splitter(splits)
    vectordb = vector_database(chunks)
    retriever = vectordb.as_retriever()
    return retriever


## QA Chain
def retriever_qa(file, query):
    llm = get_llm()
    retriever_obj = retriever(file)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False,
    )
    response = qa.invoke(query)
    return response["result"]


# Create Gradio interface
rag_application = gr.Interface(
    fn=retriever_qa,
    allow_flagging="never",
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath",
        ),  # Drag and drop file upload
        gr.Textbox(
            label="Input Query", lines=2, placeholder="Type your question here..."
        ),
    ],
    outputs=gr.Textbox(label="Output"),
    title="RAG Chatbot",
    description="Upload a PDF document and ask any question. The chatbot will try to answer using the provided document.",
)


# Launch the app
rag_application.launch(server_name="0.0.0.0", server_port=7860)

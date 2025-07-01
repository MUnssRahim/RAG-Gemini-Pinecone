import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import google.generativeai as genai
from langchain.vectorstores import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import streamlit as st
import pinecone

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment
GoogleAPIKEY = os.getenv("Google_API_KEY")
PineconeAPIKEY = os.getenv("Pinecone_API_KEY")

# Configure the genai object with the API key
genai.configure(api_key=GoogleAPIKEY)

# Initialize Pinecone
pinecone.init(api_key=PineconeAPIKEY)
index_name = "pdf-index"

def getpdftext(file):
    """Extracts text from a PDF file."""
    text = ""
    pdf_reader = PdfReader(file)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def texttochunk(text, chunk_size=10000, chunk_overlap=50):
    """Splits text into chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_text(text)
    return chunks

def getconversation(prompt_template):
    """Creates a question-answering chain."""
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'questions'])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def storetoPinecone(text_chunks):
    """Creates a local vector store and uploads to Pinecone."""
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="embedding-v1")  # Example model name
        vectorstore = Pinecone.from_texts(text_chunks, embedding=embeddings, index_name=index_name)
    except Exception as e:
        raise

def userinput(userquestion, chain):
    """Handles user input and generates a response."""
    embeddings = GoogleGenerativeAIEmbeddings(model="embedding-v1")
    try:
        vectorstore = Pinecone(index_name=index_name, embedding=embeddings)
        docs = vectorstore.similarity_search(userquestion)
        response = chain({"input_documents": docs, "question": userquestion}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error("Error with Pinecone index. Please upload a PDF file first.")

def main():
    st.set_page_config(page_title="Chat with PDF", page_icon=":book:")
    st.header("Chat using Google's Gemini")

    prompt_template = "Given the following context: {context}\nAnswer the following questions: {questions}"
    chain = getconversation(prompt_template)

    with st.sidebar:
        st.title("Menu")
        st.write("Upload your PDF file")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            try:
                with st.spinner("Please wait..."):
                    rawtext = getpdftext(uploaded_file)
                    text_chunks = texttochunk(rawtext, chunk_size=400, chunk_overlap=40)
                    storetoPinecone(text_chunks)
                    st.success("Completion")
            except Exception as e:
                st.error(f"Error occurred in uploading: {e}")

    user_question = st.text_input("Enter your question:")
    if user_question:
        userinput(user_question, chain)

if __name__ == "__main__":
    main()

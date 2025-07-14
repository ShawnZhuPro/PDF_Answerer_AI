import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import GooglePalmEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from google.oauth2 import service_account
import google.generativeai as genai
from constants import GEMINI_API_KEY

# Set up Google Cloud authentication
credentials = service_account.Credentials.from_service_account_file(
    r'C:\Users\shawn\Dev\seventh-dynamo-423823-f4-6a47c816ed48.json',
    scopes=['https://www.googleapis.com/auth/cloud-platform']
)

# Set the credentials for the current environment
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'C:\Users\shawn\Dev\seventh-dynamo-423823-f4-6a47c816ed48.json'
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

# Function to process PDF and create vector store
@st.cache_resource
def process_pdf(pdf_file):
    pdf_reader = PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = GooglePalmEmbeddings()
    
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    return knowledge_base

# Function to answer questions
def answer_question(knowledge_base, question):
    docs = knowledge_base.similarity_search(question, k=4)
    context = "\n".join([doc.page_content for doc in docs])
    
    model = genai.GenerativeModel('gemini-pro')
    prompt = f"Based on the following context, answer the question. If the answer is not in the context, say 'I don't have enough information to answer that.'\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:"
    
    response = model.generate_content(prompt)
    return response.text

# Streamlit app
st.title("PDF Question Answering")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with st.spinner("Processing PDF..."):
        knowledge_base = process_pdf(uploaded_file)
    st.success("PDF processed successfully!")

    question = st.text_input("Ask a question about the PDF:")
    if st.button("Get Answer"):
        if question:
            with st.spinner("Generating answer..."):
                answer = answer_question(knowledge_base, question)
            st.write("Answer:", answer)
        else:
            st.warning("Please enter a question.")
else:
    st.info("Please upload a PDF file to get started.")



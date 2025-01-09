import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """
    Extract text from uploaded PDF files.
    """
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
    return text

def get_text_chunks(text):
    """
    Split the extracted text into smaller chunks.
    """
    if not text.strip():
        raise ValueError("The input text is empty. Ensure the PDFs contain extractable text.")
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    
    if not chunks:
        raise ValueError("Text splitting failed. No chunks were created.")
    
    return chunks

def get_vector_store(text_chunks):
    """
    Create and save a FAISS vector store from text chunks.
    """
    if not text_chunks:
        raise ValueError("No text chunks provided to create the vector store.")

    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        raise

def get_conversational_chain():
    prompt_template = """
    You are an intelligent assistant. Use the following context to answer the question as thoroughly as possible. 
    If the answer is not in the context, explicitly say: "The answer is not available in the provided context."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


def user_input(user_question):
    """
    Handle user question and provide an answer using the conversational chain.
    """
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        
        if not docs:
            st.warning("No relevant context found for the question.")
            return

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply:", response["output_text"])
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

def main():
    st.set_page_config(page_title="Chat with PDF", layout="wide")
    st.header("Chat with PDF using Gemini 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings)
        docs = new_db.similarity_search(user_question)
        st.write("Retrieved Documents:", docs)  # Debug retrieved documents

        if docs:
            chain = get_conversational_chain()
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("Reply:", response["output_text"])
        else:
            st.warning("No relevant context found for the question.")

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    try:
                        # Step 1: Extract text
                        raw_text = get_pdf_text(pdf_docs)
                        st.write("Extracted Text Preview:", raw_text[:1000])  # Debug extracted text

                        # Step 2: Split text into chunks
                        text_chunks = get_text_chunks(raw_text)
                        st.write("Sample Chunks:", text_chunks[:5])  # Debug text chunks

                        # Step 3: Create vector store
                        get_vector_store(text_chunks)
                        st.success("Files processed successfully!")
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
            else:
                st.warning("Please upload PDF files before processing.")


if __name__ == "__main__":
    main()

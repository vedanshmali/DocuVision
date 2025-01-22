import os
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading PDF file: {str(e)}")
    return text

def get_excel_text(excel_files):
    text = ""
    excel_data = {}
    for excel in excel_files:
        try:
            df = pd.read_excel(excel)
            text += df.to_string(index=False)
            excel_data[excel.name] = df  # Store Excel data for person details feature
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
    return text, excel_data

def get_text_chunks(text):
    if not text.strip():
        raise ValueError("The input text is empty. Ensure the files contain extractable text.")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    if not chunks:
        raise ValueError("Text splitting failed. No chunks were created.")
    return chunks

def get_vector_store(text_chunks):
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

def chat_with_user(question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(question)

        if not docs:
            return "No relevant context found for the question."

        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"Error generating response: {str(e)}"

def get_summary(text):
    return "Summary: " + " ".join(text.split()[:100]) + "..."

def get_person_details(person_name, excel_data):
    details = []
    for file, df in excel_data.items():
        matched_rows = df[df.apply(lambda row: row.astype(str).str.contains(person_name, case=False).any(), axis=1)]
        if not matched_rows.empty:
            details.append((file, matched_rows))
    return details

def visualize_excel_data(excel_data):
    for file, df in excel_data.items():
        st.subheader(f"Data from {file}")
        st.write(df)

        # Plot data if possible (only for numerical data)
        if df.select_dtypes(include=['number']).shape[1] > 0:
            st.subheader("Data Visualization")
            numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
            st.write(f"Visualizing data for: {', '.join(numerical_columns)}")

            for col in numerical_columns:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

def main():
    st.set_page_config(page_title="Chat with Files", layout="wide")
    st.title("Chat with PDF and Excel Files")

    # Custom Dark Theme CSS
    st.markdown(
        """
        <style>
        body {
            background-color: #2e2e2e;
            color: #f1f1f1;
        }
        .main {
            background-color: #2e2e2e;
            padding: 20px;
        }
        .sidebar {
            background-color: #383838;
            border-right: 2px solid #505050;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #f1f1f1;
            font-family: 'Arial', sans-serif;
        }
        .stButton button {
            background-color: #1c6b91;
            color: white;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #165a7f;
        }
        .stTextInput input {
            background-color: #383838;
            border: 2px solid #1c6b91;
            padding: 10px;
            border-radius: 5px;
            font-size: 14px;
            color: #f1f1f1;
        }
        .stTextInput input:focus {
            outline: none;
            border-color: #1c6b91;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Sidebar for file uploads
    st.sidebar.header("Upload Files for Processing")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
    excel_files = st.sidebar.file_uploader("Upload Excel Files", accept_multiple_files=True, type=["xlsx"])
    process_button = st.sidebar.button("Process Files")

    user_question = st.text_input("Ask a question based on uploaded files:")
    person_name = st.text_input("Enter person's name to get details:")

    # Initialize raw_text variable
    raw_text = ""
    excel_data = {}

    if process_button:
        if pdf_docs or excel_files:
            with st.spinner("Processing..."):
                try:
                    if pdf_docs:
                        raw_text += get_pdf_text(pdf_docs)
                    if excel_files:
                        raw_text, excel_data = get_excel_text(excel_files)

                    if raw_text.strip():  # Only process if text is not empty
                        st.write("Extracted Text (Debug):", raw_text[:1000])  # Debug text extraction

                        text_chunks = get_text_chunks(raw_text)
                        st.write("Text Chunks (Debug):", text_chunks[:5])  # Debug text chunks

                        get_vector_store(text_chunks)
                        st.success("Files processed successfully! You can now ask questions.")
                    else:
                        st.warning("No extractable text found in the uploaded files.")
                except Exception as e:
                    st.error(f"Error during processing: {str(e)}")
        else:
            st.warning("Please upload files before processing.")

    # Person details feature
    if person_name:
        st.write(f"Searching for details about: {person_name}")
        details = get_person_details(person_name, excel_data)
        if details:
            for file, data in details:
                st.write(f"Details from {file}:")
                st.write(data)
        else:
            st.write(f"No details found for {person_name}.")

    # Question-answer feature
    if user_question:
        st.write("Your Question:", user_question)
        answer = chat_with_user(user_question)
        st.write("Answer:", answer)

    # Summary feature
    if raw_text.strip():
        summary = get_summary(raw_text)
        st.write("Summary:", summary)

    # Word count feature
    if raw_text.strip():
        st.write("Word count of the extracted text:", len(raw_text.split()))

    # Visualize Excel Data (Charts)
    if excel_data:
        visualize_excel_data(excel_data)

if __name__ == "__main__":
    main()

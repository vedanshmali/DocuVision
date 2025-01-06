# DocuVision
 DocuVision: Intelligent Document Processing DocuVision is a cutting-edge tool designed to revolutionize document analysis and question answering by leveraging state-of-the-art AI technologies. It seamlessly integrates Optical Character Recognition (OCR), natural language processing.
Steps to Set Up and Run DocuVision
Follow these steps to set up and run the DocuVision application:



Requirements
Python 3.7+

Libraries:

streamlit
PyPDF2
pandas
Pillow
pytesseract
langchain
faiss-cpu
langchain-google-genai
google-generativeai
python-dotenv

Steps to Set Up and Run DocuVision
Follow these steps to set up and run the DocuVision application:

Step 1: Clone the Repository
Download the project files by cloning the repository:
bash
Copy code
git clone https://github.com/yourusername/DocuVision.git
cd DocuVision

Step 2: Install Python
Ensure you have Python 3.7 or higher installed. You can check your Python version using:
bash
Copy code
python --version
If not installed, download and install Python from python.org.

Step 3: Install Dependencies
Install the required Python libraries listed in requirements.txt:
bash
Copy code
pip install -r requirements.txt

Step 4: Install Tesseract OCR
For Linux:
Run the following command:
bash
Copy code
sudo apt install tesseract-ocr
For Windows:
Download Tesseract from Tesseract OCR GitHub.
Install it and note the installation path (e.g., C:\Program Files\Tesseract-OCR).
Update the pytesseract configuration in app.py:
python
Copy code
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

Step 5: Configure Environment Variables
Create a .env file in the project directory.
Add your Google API Key:
plaintext
Copy code
GOOGLE_API_KEY=your_google_api_key_here
Replace your_google_api_key_here with your actual API key.

Step 6: Run the Application
Launch the application using Streamlit:
bash
Copy code
streamlit run app.py

Step 7: Use the Application
Open the app in your browser (it will automatically launch, or go to http://localhost:8501).
Upload Files: Drag and drop PDFs, Excel files, or images into the file uploader.
Process Files: Click the "Process Files" button to extract and analyze the document data.
Ask Questions: Enter a question in the input field to receive context-aware answers.
Requirements Summary
Python 3.7+
Dependencies:
streamlit, PyPDF2, pandas, Pillow, pytesseract, langchain, faiss-cpu, langchain-google-genai, google-generativeai, python-dotenv
Tesseract OCR
Google API Key
These steps will ensure you can set up and run DocuVision smoothly on your system! 🚀

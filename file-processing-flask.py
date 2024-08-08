import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import requests
import json
import pandas as pd
import pdfplumber
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from PIL import Image
import pytesseract
from groq import Groq
import firebase_admin
from firebase_admin import credentials, firestore, storage
from io import BytesIO

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path as needed
GROQ_API_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_API_KEY = "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"

# Initialize Groq client
groq_client = Groq(api_key=GROQ_API_KEY)

SUPPORTED_FILE_TYPES = ['.pdf', '.xlsx', '.xls', '.csv', '.txt', '.docx', '.jpg', '.jpeg', '.png', '.wav', '.mp3']

# Initialize Firebase Admin SDK
cred = credentials.Certificate(r"D:\Flask\train-setai-firebase-adminsdk-3rr20-dc2b855d99.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'train-setai.appspot.com'  # Correct bucket name format
})
db = firestore.client()
bucket = storage.bucket()

# Define the path where you want to store the final datasets
LOCAL_DATASET_DIR = r"D:\Flask\datasets"

def save_final_dataset(filename, final_dataset):
    try:
        # Save to Firestore
        doc_ref = db.collection('final_datasets').document(filename)
        doc_ref.set({'dataset': final_dataset})
        print(f"Saved final dataset to Firestore for file: {filename}")

        # Ensure the local directory exists
        os.makedirs(LOCAL_DATASET_DIR, exist_ok=True)

        # Save to local file
        local_file_path = os.path.join(LOCAL_DATASET_DIR, f'{os.path.splitext(filename)[0]}_dataset.json')
        with open(local_file_path, 'w') as f:
            json.dump(final_dataset, f, indent=4)
        print(f"Saved final dataset to local directory: {local_file_path}")

    except Exception as e:
        print(f"Exception in save_final_dataset: {str(e)}")
        raise e

@app.route('/')
def upload_file():
    return render_template('upload.html')

@app.route('/process', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_type = determine_file_type(file.filename)
    if file_type not in SUPPORTED_FILE_TYPES:
        return jsonify({'error': 'Unsupported file type'}), 400
    
    try:
        print("Uploading raw file to Firebase Storage...")
        raw_file_url = upload_to_firebase_storage(file, file.filename, 'raw_user_files')
        print(f"Raw file uploaded. URL: {raw_file_url}")

        print(f"Processing file of type: {file_type}...")
        processed_text = process_file_by_type(file, file_type)
        print("File processed successfully.")

        processed_txt_filename = os.path.splitext(file.filename)[0] + '.txt'
        processed_txt_file = save_data_to_memory(processed_text)
        print("Uploading processed .txt file to Firebase Storage...")
        processed_file_url = upload_to_firebase_storage(processed_txt_file, processed_txt_filename, 'processed_files')
        print(f"Processed .txt file uploaded. URL: {processed_file_url}")

        print("Generating final dataset...")
        final_dataset = generate_instructions_and_process(processed_text)
        print("Final dataset generated.")

        print("Saving final dataset to Firestore...")
        save_final_dataset(file.filename, final_dataset)
        print("Final dataset saved to Firestore.")

        print("Saving metadata to Firestore...")
        save_metadata_to_firebase(file.filename, file_type, raw_file_url, processed_file_url, final_dataset)
        print("Metadata saved to Firestore.")

        return jsonify({'message': 'File processed successfully', 'file_type': file_type, 'data': final_dataset})
    except Exception as e:
        print(f"Exception occurred: {str(e)}")
        return jsonify({'error': str(e)}), 500

def determine_file_type(filename):
    file_ext = os.path.splitext(filename)[1].lower()
    print(f"Determined file type: {file_ext}")
    return file_ext

def extract_text_from_pdf(pdf_file):
    try:
        with pdfplumber.open(BytesIO(pdf_file.read())) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
        print("Extracted text from PDF.")
        return text
    except Exception as e:
        print(f"Exception in extract_text_from_pdf: {str(e)}")
        raise e

def load_excel(excel_file):
    try:
        df = pd.read_excel(BytesIO(excel_file.read()))
        print("Loaded Excel file.")
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Exception in load_excel: {str(e)}")
        raise e

def load_csv(csv_file):
    try:
        df = pd.read_csv(BytesIO(csv_file.read()))
        print("Loaded CSV file.")
        return df.to_csv(index=False)
    except Exception as e:
        print(f"Exception in load_csv: {str(e)}")
        raise e

def read_text_file(text_file):
    try:
        text = text_file.read().decode('utf-8')
        print("Read text file.")
        return text
    except Exception as e:
        print(f"Exception in read_text_file: {str(e)}")
        raise e

def extract_text_from_docx(docx_file):
    try:
        doc = DocxDocument(BytesIO(docx_file.read()))
        text = ""
        for para in doc.paragraphs:
            text += para.text + "\n"
        print("Extracted text from DOCX file.")
        return text
    except Exception as e:
        print(f"Exception in extract_text_from_docx: {str(e)}")
        raise e

def extract_text_from_image(image_file):
    try:
        image_bytes = BytesIO(image_file.read())
        img = Image.open(image_bytes)
        text = pytesseract.image_to_string(img, config='--psm 6')
        print("Extracted text from image file.")
        return text
    except Exception as e:
        print(f"Exception in extract_text_from_image: {str(e)}")
        raise e

def transcribe_audio_with_groq(audio_file):
    try:
        audio_bytes = BytesIO(audio_file.read())
        audio_bytes.seek(0)
        headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
        model = "whisper-large-v3"
        response = requests.post(GROQ_API_URL, headers=headers, files={'file': audio_bytes}, data={'model': model})
        
        if response.status_code == 200:
            print("Transcribed audio with Groq.")
            return response.json().get('text', 'Transcription failed')
        else:
            print(f"Error transcribing audio with Groq: {response.status_code} - {response.text}")
            return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        print(f"Exception in transcribe_audio_with_groq: {str(e)}")
        raise e

def process_file_by_type(file, file_ext):
    file.seek(0)  # Ensure the file pointer is at the beginning
    try:
        if file_ext == '.pdf':
            return extract_text_from_pdf(file)
        elif file_ext in ['.xlsx', '.xls']:
            return load_excel(file)
        elif file_ext == '.csv':
            return load_csv(file)
        elif file_ext == '.txt':
            return read_text_file(file)
        elif file_ext == '.docx':
            return extract_text_from_docx(file)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            return extract_text_from_image(file)
        elif file_ext in ['.wav', '.mp3']:
            return transcribe_audio_with_groq(file)
        else:
            raise Exception('Unsupported file type')
    except Exception as e:
        print(f"Exception in process_file_by_type: {str(e)}")
        raise e

def save_data_to_memory(text):
    try:
        memory_file = BytesIO()
        memory_file.write(text.encode('utf-8'))
        memory_file.seek(0)
        print("Saved data to memory.")
        return memory_file
    except Exception as e:
        print(f"Exception in save_data_to_memory: {str(e)}")
        raise e

def generate_instructions_and_process(text):
    try:
        chunks = chunk_text(text, 3000)
        dataset = []

        for chunk in chunks:
            instructions = generate_instructions(chunk)
            for instruction in instructions:
                result = process_instruction(chunk, instruction)
                if result:
                    dataset.append(result)
        print("Generated instructions and processed text.")
        return dataset
    except Exception as e:
        print(f"Exception in generate_instructions_and_process: {str(e)}")
        raise e

def chunk_text(text, max_length):
    print("Chunking text.")
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def generate_instructions(chunk):
    try:
        instruction_request = f"Generate instructions that a user could ask about the following context:\n\n{chunk}\n\nProvide each instruction in 2-3 sentences on a new line."
        
        messages = [
            {"role": "system", "content": "You are an assistant that generates plausible instructions for a given context."},
            {"role": "user", "content": instruction_request}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192", 
            temperature=0.7,
            max_tokens=500,
            top_p=1,
            stop=None,
            stream=False
        )

        response_content = chat_completion.choices[0].message.content.strip()
        instructions = [instr.strip() for instr in response_content.split("\n") if instr]
        print("Generated instructions.")
        return instructions
    except Exception as e:
        print(f"Exception in generate_instructions: {str(e)}")
        raise e

def process_instruction(chunk, instruction):
    try:
        messages = [
            {"role": "system", "content": "You are an assistant that generates concise answers based on the provided instruction and context."},
            {"role": "user", "content": f"Instruction: {instruction}\nContext: {chunk}\n\nAnswer in 2-3 sentences."}
        ]

        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192", 
            temperature=0.6,
            max_tokens=200,
            top_p=1,
            stop=None,
            stream=False
        )

        response_content = chat_completion.choices[0].message.content.strip()
        result = {
            "instruction": instruction,
            "context": chunk,
            "answer": response_content
        }
        print("Processed instruction.")
        return result
    except Exception as e:
        print(f"Exception in process_instruction: {str(e)}")
        raise e

def upload_to_firebase_storage(file, filename, folder):
    try:
        blob = bucket.blob(f'{folder}/{filename}')
        blob.upload_from_file(file)
        blob.make_public()
        print(f"Uploaded file to Firebase Storage: {blob.public_url}")
        return blob.public_url
    except Exception as e:
        print(f"Exception in upload_to_firebase_storage: {str(e)}")
        raise e

def save_metadata_to_firebase(filename, file_type, raw_file_url, processed_file_url, final_dataset):
    try:
        metadata = {
            'file_type': file_type,
            'raw_file_url': raw_file_url,
            'processed_file_url': processed_file_url,
            'final_dataset': final_dataset
        }
        doc_ref = db.collection('metadata').document(filename)
        doc_ref.set(metadata)
        print(f"Saved metadata for file: {filename}")
    except Exception as e:
        print(f"Exception in save_metadata_to_firebase: {str(e)}")
        raise e

if __name__ == '__main__':
    app.run(debug=True)



# File Processing and Analysis with Flask

This repository provides a Flask application for uploading and processing various file types. The application extracts text from documents, images, and audio files, then stores and processes the data using Groq, Firebase, and LLaMA. The processed data is saved both to Firebase Firestore and locally.

## Features

- **Upload and Process Files**: Supports PDFs, Excel, CSV, text, DOCX, images (JPG, JPEG, PNG), and audio files (WAV, MP3).
- **Text Extraction**: Extracts text from documents and images using OCR.
- **Audio Transcription**: Transcribes audio files using Groq.
- **Data Processing with LLaMA**: Uses the LLaMA model for generating instructions and processing text.
- **Data Storage**: Saves processed data to Firebase Firestore and Firebase Storage.
- **Local Dataset Storage**: Saves final datasets locally.

## Requirements

- Python 3.7+
- Flask
- Flask-CORS
- Requests
- Pandas
- pdfplumber
- PyPDF2
- python-docx
- Pillow
- pytesseract
- Groq Python Client
- Firebase Admin SDK
- Transformers library (for LLaMA)

Install the required libraries using:

```bash
pip install flask flask-cors requests pandas pdfplumber PyPDF2 python-docx Pillow pytesseract groq firebase-admin transformers
```

## Configuration

1. **Groq API Key**: Update `GROQ_API_KEY` with your Groq API key in `app.py`.

2. **Firebase Admin SDK**: Update the path to your Firebase Admin SDK JSON file in `app.py`.

3. **Tesseract Path**: Update the `pytesseract.pytesseract.tesseract_cmd` path if Tesseract OCR is installed in a different location on your system.

4. **LLaMA Model Configuration**: Ensure the LLaMA model configuration in your code is correct. Update the model parameters as needed in `app.py`.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Nimra3261/file-processing-flask.git
   cd file-processing-flask
   ```

2. **Create a Virtual Environment (Optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Application:**

   ```bash
   python app.py
   ```

   The Flask application will start and be available at `http://127.0.0.1:5000`.

## Usage

1. **Upload a File:**

   - Navigate to `http://127.0.0.1:5000/` in your web browser.
   - Use the upload form to select and upload a file.

2. **Processing:**

   - The application will process the file based on its type and extract relevant data.
   - The processed file and its metadata will be saved to Firebase Storage and Firestore.
   - The LLaMA model will be used to generate instructions and process the text.

## API Endpoints

- **GET /**: Displays the upload form.

- **POST /process**: Processes the uploaded file. The response includes the processed data and URLs for the raw and processed files.

## Error Handling

- Errors encountered during file processing or data storage are logged and returned as part of the API response.

## Contributing

Contributions are welcome! Please open issues or submit pull requests with improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- Groq API for audio transcription.
- Firebase for cloud storage and database management.
- Tesseract OCR for text extraction from images.
- LLaMA model for generating instructions and processing text.

---


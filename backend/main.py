import os
import shutil
import tempfile
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
# import textract
import pymongo
import pandas as pd
import docx

# Load environment variables from .env file (if any)
load_dotenv()

origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:3000",
    "http://localhost:8000/predict"
]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Response(BaseModel):
    result: str | None

# MongoDB Configuration
MONGO_URI = os.getenv("MONGO_URI")
client = pymongo.MongoClient(MONGO_URI)
db = client["file_storage"]

# Load the DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

# Load the DistilBERT model for question answering
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Function to perform question answering using DistilBERT
def answer_question(context: str, question: str) -> str:
    inputs = tokenizer(question, context, return_tensors="pt", max_length=512, truncation=True)
    outputs = model(**inputs)
    
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    
    print("Start Logits:", start_logits)
    print("End Logits:", end_logits)
    
    if isinstance(start_logits, torch.Tensor) and isinstance(end_logits, torch.Tensor):
        start_index = torch.argmax(start_logits)
        end_index = torch.argmax(end_logits)
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index+1]))
    else:
        answer = "Error: Invalid model output"
    
    return answer

# Function to extract text from different file types
def extract_text(file_path: str) -> str:
    file_ext = os.path.splitext(file_path)[-1].lower()
    if file_ext == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    elif file_ext == ".csv":
        # Assume the first column of the CSV contains text data
        df = pd.read_csv(file_path)
        text = " ".join(df.iloc[:, 0].astype(str).tolist())
    elif file_ext == ".docx":
        doc = docx.Document(file_path)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    # elif file_ext == ".pdf":
    #     text = textract.process(file_path).decode("utf-8")
    else:
        text = ""
    return text



@app.post("/predict", response_model=Response)
def predict_text(file: UploadFile = File(...), question: str = Form(...)) -> Any:
    # Create a temporary directory to store the uploaded file
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)
    
    # Save the uploaded file
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    
    # Extract text from the uploaded file
    context = extract_text(file_path)
    
    # Perform question answering
    if context and question:
        result = answer_question(context, question)
        # print(result)
    else:
        result = None
    
    # Store file information in MongoDB
    file_info = {
        "filename": file.filename,
        "filepath": file_path,
        "question": question,
        "context": context,
        "result": result
    }
    db.files.insert_one(file_info)

    # Delete the temporary directory
    shutil.rmtree(temp_dir)

    return {"result": result}

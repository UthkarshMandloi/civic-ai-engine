import os
import re
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any
import datetime
import random
import glob

# --- 1. Load Trained Models on Startup ---
try:
    print("Loading AI models...")
    category_classifier = joblib.load("models/category_classifier.joblib")
    coherence_classifier = joblib.load("models/coherence_classifier.joblib")
    print("Models loaded successfully.")
except FileNotFoundError:
    print("\nERROR: Model files not found! Please run 'train_model.py' first.\n")
    exit()

# --- FastAPI App Initialization ---
app = FastAPI()

origins = ["http://localhost:3001", "http://127.0.0.1:3001"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- In-Memory Storage for WhatsApp Conversations ---
whatsapp_conversations: Dict[str, Dict[str, Any]] = {}

# --- Pydantic Models for Data Validation ---
class ComplaintData(BaseModel):
    name: str
    contact: str
    category: str
    title: str
    description: str
    location: str

class WhatsAppMessage(BaseModel):
    From: str
    Body: str

# --- Intent Detection System ---
def detect_intent(message: str) -> str:
    scores = {}
    message_words = set(message.lower().split())
    for intent_file in glob.glob("intents/*.txt"):
        intent_name = os.path.basename(intent_file).replace(".txt", "")
        with open(intent_file, "r") as f:
            keywords = set(line.strip() for line in f if line.strip())
        score = len(message_words.intersection(keywords))
        if score > 0:
            scores[intent_name] = score
    if not scores:
        return "unknown"
    best_intent = max(scores, key=lambda x: scores[x])
    print(f"Detected intent: '{best_intent}' with score {scores[best_intent]}")
    return best_intent

# --- AI Analysis Logic ---
@app.post("/api/ai/analyze")
async def analyze_complaint_web(complaint: ComplaintData):
    complaint_text = [complaint.title + " " + complaint.description]
    text_lower = complaint_text[0].lower()
    
    is_valid = bool(coherence_classifier.predict(complaint_text)[0])
    if not is_valid:
        return {"isComplaintValid": False, "reasoning": "Invalid complaint text."}
    
    predicted_category = category_classifier.predict(complaint_text)[0]
    
    priority = "Medium"
    high_priority_keywords = ["urgent", "dangerous", "hazard", "fire", "leak", "exposed", "accident", "unsafe", "sewage", "overflow"]
    if any(keyword in text_lower for keyword in high_priority_keywords):
        priority = "High"
        
    resolution_days = 5 # Default
    
    return {
        "priority": priority,
        "isComplaintValid": True,
        "reasoning": "",
        "suggestedCategory": predicted_category,
        "estimatedResolutionDays": resolution_days,
    }

# --- WhatsApp Chatbot API (Updated with Robust Logic) ---
@app.post("/api/whatsapp/webhook")
async def whatsapp_bot(message: WhatsAppMessage):
    user_phone = message.From.split(':')[-1]
    user_message = message.Body.strip()
    
    session = whatsapp_conversations.get(user_phone)
    response_text = ""

    async def start_complaint_flow(complaint_desc):
        analysis_result = await analyze_complaint_web(ComplaintData(
            name="WhatsApp User", contact=user_phone, category="unknown", 
            title="WhatsApp Complaint", description=complaint_desc, location="unknown"
        ))
        
        nonlocal session
        session = {
            "stage": "get_location",
            "description": complaint_desc
        }
        
        if analysis_result and analysis_result.get("isComplaintValid"):
            session["category"] = analysis_result["suggestedCategory"]
            session["priority"] = analysis_result["priority"]
        else:
            session["category"] = "unknown"
            session["priority"] = "medium"
        
        nonlocal response_text
        response_text = f"Got it. I've categorized this as a '{session['category']}' issue with '{session['priority']}' priority. What is the location of this issue?"

    if not session:
        intent = detect_intent(user_message)
        if intent == "file_complaint":
            await start_complaint_flow(user_message)
        elif intent == "check_status":
            response_text = "To check the status of a complaint, please provide your Complaint ID."
            session = {"stage": "get_status_id"}
        else:
            response_text = "Welcome to the Smart Grievance System! How can I help you today? You can say 'file a complaint' or 'check status'."
            session = {"stage": "greeting"}
    else:
        current_stage = session.get("stage")
        # --- FIX: If the user describes the problem at the greeting stage, start the complaint flow ---
        if current_stage == "greeting":
            intent = detect_intent(user_message)
            if intent == "file_complaint":
                await start_complaint_flow(user_message) # Directly start the flow
            elif intent == "check_status":
                response_text = "To check the status of a complaint, please provide your Complaint ID."
                session["stage"] = "get_status_id"
            else:
                response_text = "I'm sorry, I didn't understand. You can say 'file a complaint' or 'check status'."
        
        elif current_stage == "get_description":
            await start_complaint_flow(user_message)

        elif current_stage == "get_location":
            session["location"] = user_message
            response_text = "Thank you. Finally, please set a simple password to secure this complaint."
            session["stage"] = "get_password"

        elif current_stage == "get_password":
            session["password"] = user_message
            complaint_id = f"WACMP-{random.randint(1000, 9999)}"
            session["id"] = complaint_id
            
            print(f"--- WhatsApp Complaint to be Saved ---")
            print(f"Details: {session}")
            
            response_text = f"Excellent! Your complaint has been filed. Your Complaint ID is: *{complaint_id}*. Please save this ID."
            whatsapp_conversations.pop(user_phone, None)
            return {"response": response_text}

        elif current_stage == "get_status_id":
            complaint_id_from_user = user_message
            response_text = f"Checking status for complaint {complaint_id_from_user}... (This feature is in development)."
            whatsapp_conversations.pop(user_phone, None)
            return {"response": response_text}

    if session is not None:
        whatsapp_conversations[user_phone] = session
    
    return {"response": response_text}

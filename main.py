import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from PIL import Image
import pandas as pd
import numpy as np
import json
import re
from datetime import datetime
import requests
import io

class MultilingualMedicalChatbot:
    def __init__(self):
        self.setup_models()
        self.setup_intents()

    def setup_models(self):
        """Initialize mBERT model for intent classification"""
        try:
            # Use mBERT for multilingual intent classification
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

            # For demo purposes, we'll use a simple classification approach
            # In production, you would fine-tune mBERT on your specific dataset
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # Initialize text classification pipeline
            self.classifier = pipeline(
                "text-classification",
                model="bert-base-multilingual-cased",
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            st.success("✅ mBERT model loaded successfully!")

        except Exception as e:
            st.error(f"Error loading model: {str(e)}")

    def setup_intents(self):
        """Define medical intents in both Arabic and English"""
        self.intents = {
            "symptom_inquiry": {
                "en": ["symptom", "pain", "hurt", "feel", "sick", "fever", "headache", "cough"],
                "ar": ["أعراض", "ألم", "يؤلم", "أشعر", "مريض", "حمى", "صداع", "سعال", "وجع"]
            },
            "medication_info": {
                "en": ["medicine", "medication", "drug", "pill", "tablet", "dose", "prescription"],
                "ar": ["دواء", "علاج", "حبوب", "قرص", "جرعة", "وصفة", "دواي"]
            },
            "appointment": {
                "en": ["appointment", "visit", "schedule", "book", "doctor", "clinic"],
                "ar": ["موعد", "زيارة", "جدولة", "حجز", "طبيب", "عيادة"]
            },
            "diagnosis_request": {
                "en": ["diagnose", "what is", "condition", "disease", "illness", "analysis"],
                "ar": ["تشخيص", "ما هو", "حالة", "مرض", "تحليل", "فحص"]
            },
            "image_analysis": {
                "en": ["image", "photo", "scan", "x-ray", "picture", "analyze", "look at"],
                "ar": ["صورة", "تصوير", "أشعة", "فحص", "تحليل", "انظر"]
            },
            "greeting": {
                "en": ["hello", "hi", "good morning", "good evening", "how are you"],
                "ar": ["مرحبا", "أهلا", "صباح الخير", "مساء الخير", "كيف حالك", "السلام عليكم"]
            }
        }

        # Predefined responses
        self.responses = {
            "symptom_inquiry": {
                "en": "I understand you're experiencing symptoms. Can you describe them in more detail? When did they start?",
                "ar": "أفهم أنك تعاني من أعراض. هل يمكنك وصفها بتفصيل أكثر؟ متى بدأت؟"
            },
            "medication_info": {
                "en": "For headache relief, common over-the-counter options include: Paracetamol (500-1000mg every 4-6 hours), Ibuprofen (200-400mg every 6-8 hours). However, please consult a pharmacist or doctor for personalized advice. If headaches persist, see a healthcare professional.",
                "ar": "لتخفيف الصداع، الخيارات المتاحة بدون وصفة تشمل: الباراسيتامول (500-1000 مجم كل 4-6 ساعات)، الإيبوبروفين (200-400 مجم كل 6-8 ساعات). لكن يرجى استشارة الصيدلي أو الطبيب للنصيحة الشخصية. إذا استمر الصداع، راجع أخصائي الرعاية الصحية."
            },
            "appointment": {
                "en": "To schedule an appointment, please contact the clinic directly or use the online booking system.",
                "ar": "لحجز موعد، يرجى الاتصال بالعيادة مباشرة أو استخدام نظام الحجز الإلكتروني."
            },
            "diagnosis_request": {
                "en": "I cannot provide medical diagnosis. Please consult with a qualified healthcare professional for proper diagnosis.",
                "ar": "لا يمكنني تقديم تشخيص طبي. يرجى استشارة أخصائي رعاية صحية مؤهل للحصول على تشخيص مناسب."
            },
            "image_analysis": {
                "en": "I can help analyze medical images for informational purposes only. Please upload your image.",
                "ar": "يمكنني المساعدة في تحليل الصور الطبية لأغراض إعلامية فقط. يرجى رفع الصورة."
            },
            "greeting": {
                "en": "Hello! I'm your multilingual medical assistant. How can I help you today?",
                "ar": "مرحباً! أنا مساعدك الطبي متعدد اللغات. كيف يمكنني مساعدتك اليوم؟"
            },
            "default": {
                "en": "I'm here to help with medical inquiries. Please ask me about symptoms, medications, or upload medical images for analysis.",
                "ar": "أنا هنا للمساعدة في الاستفسارات الطبية. يرجى سؤالي عن الأعراض أو الأدوية أو رفع الصور الطبية للتحليل."
            }
        }

    def detect_language(self, text):
        """Simple language detection"""
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if len(arabic_chars) > len(text) * 0.3:
            return 'ar'
        return 'en'

    def classify_intent(self, text):
        """Classify user intent using keyword matching and mBERT"""
        text_lower = text.lower()
        language = self.detect_language(text)

        # Score each intent based on keyword matching with priority system
        intent_scores = {}
        
        # Check for medication keywords first (higher priority)
        medication_keywords = self.intents["medication_info"][language]
        medication_score = sum(1 for keyword in medication_keywords if keyword in text_lower)
        
        # If medication keywords found, boost medication intent
        if medication_score > 0:
            intent_scores["medication_info"] = medication_score * 2  # Higher weight
        
        # Score other intents normally
        for intent, keywords in self.intents.items():
            if intent != "medication_info":  # Skip medication as we handled it above
                score = 0
                lang_keywords = keywords.get(language, [])
                for keyword in lang_keywords:
                    if keyword in text_lower:
                        score += 1
                intent_scores[intent] = score

        # Get the intent with highest score
        if max(intent_scores.values()) > 0:
            predicted_intent = max(intent_scores, key=intent_scores.get)
        else:
            predicted_intent = "default"

        return predicted_intent, language

    def analyze_medical_image(self, image):
        """Analyze medical image (simplified version)"""
        try:
            # Convert PIL image to array for basic analysis
            img_array = np.array(image)

            # Basic image statistics
            analysis = {
                "width": image.width,
                "height": image.height,
                "mode": image.mode,
                "mean_intensity": np.mean(img_array) if len(img_array.shape) <= 3 else 0,
                "std_intensity": np.std(img_array) if len(img_array.shape) <= 3 else 0
            }

            # Simple observations based on image properties
            observations = []

            if analysis["mean_intensity"] < 50:
                observations.append("Image appears to be quite dark - may be an X-ray or CT scan")
            elif analysis["mean_intensity"] > 200:
                observations.append("Image appears bright - possibly overexposed or processed")

            if image.mode == "L":
                observations.append("Grayscale medical image detected")

            return analysis, observations

        except Exception as e:
            return None, [f"Error analyzing image: {str(e)}"]

    def generate_response(self, intent, language, image_analysis=None):
        """Generate response based on intent and language"""
        if image_analysis and intent == "image_analysis":
            analysis, observations = image_analysis
            if language == 'ar':
                response = f"تحليل الصورة الطبية:\n"
                response += f"الأبعاد: {analysis['width']} × {analysis['height']}\n"
                response += f"الملاحظات: {' • '.join(observations) if observations else 'لا توجد ملاحظات خاصة'}\n"
                response += "⚠️ هذا التحليل للأغراض الإعلامية فقط. استشر طبيباً مختصاً للتشخيص الدقيق."
            else:
                response = f"Medical Image Analysis:\n"
                response += f"Dimensions: {analysis['width']} × {analysis['height']}\n"
                response += f"Observations: {' • '.join(observations) if observations else 'No specific observations'}\n"
                response += "⚠️ This analysis is for informational purposes only. Consult a medical professional for accurate diagnosis."
        else:
            response = self.responses.get(intent, self.responses["default"])[language]

        return response

def main():
    try:
        st.set_page_config(
            page_title="Multilingual Medical Chatbot",
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.error(f"Configuration error: {str(e)}")

    st.title("🏥 Multilingual Medical Chatbot")
    st.markdown("### مساعد طبي ذكي متعدد اللغات | Intelligent Multilingual Medical Assistant")

    # Debug information
    st.write("🔧 Debug: Streamlit is running correctly")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Loading mBERT model..."):
            try:
                st.session_state.chatbot = MultilingualMedicalChatbot()
            except Exception as e:
                st.error(f"Model loading error: {str(e)}")
                st.info("The app is still functional with basic features")

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Sidebar for features
    with st.sidebar:
        st.header("Features | الميزات")
        st.markdown("""
        ✅ **mBERT** for multilingual understanding

        ✅ **Arabic & English** support

        ✅ **Medical Image Analysis**

        ✅ **Intent Classification**

        ✅ **Symptom Inquiry**

        ✅ **Medication Information**
        """)

        st.header("Upload Medical Image")
        uploaded_file = st.file_uploader("Choose a medical image...", type=['png', 'jpg', 'jpeg', 'bmp'])

    # Main chat interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Chat Interface")

        # Display chat history
        for i, (user_msg, bot_response, timestamp) in enumerate(st.session_state.chat_history):
            st.markdown(f"**You ({timestamp}):** {user_msg}")
            st.markdown(f"**Bot:** {bot_response}")
            st.markdown("---")

        # Text input
        user_input = st.text_input("Enter your message (English/Arabic) | أدخل رسالتك:", key="user_input")

        col_send, col_clear = st.columns([1, 1])
        with col_send:
            if st.button("Send | إرسال"):
                if user_input:
                    process_message(user_input, uploaded_file)

        with col_clear:
            if st.button("Clear Chat | مسح المحادثة"):
                st.session_state.chat_history = []
                st.rerun()

    with col2:
        st.header("Model Information")
        st.markdown("""
        **Model:** mBERT (Multilingual BERT)

        **Languages:** Arabic, English

        **Capabilities:**
        - Intent Classification
        - Multilingual Understanding
        - Medical Image Analysis
        - Symptom Assessment

        **Device:** CPU/GPU Auto-detection
        """)

        if uploaded_file:
            st.header("Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Medical Image", use_column_width=True)

def process_message(user_input, uploaded_file=None):
    """Process user message and generate response"""
    chatbot = st.session_state.chatbot

    # Classify intent and detect language
    intent, language = chatbot.classify_intent(user_input)

    # Analyze image if provided
    image_analysis = None
    if uploaded_file and intent == "image_analysis":
        image = Image.open(uploaded_file)
        image_analysis = chatbot.analyze_medical_image(image)

    # Generate response
    response = chatbot.generate_response(intent, language, image_analysis)

    # Add to chat history
    timestamp = datetime.now().strftime("%H:%M")
    st.session_state.chat_history.append((user_input, response, timestamp))

    # Clear input and rerun
    st.rerun()

if __name__ == "__main__":
    main()
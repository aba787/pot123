
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
import easyocr
import cv2
from typing import Dict, List, Tuple, Optional

class DrugAPIHandler:
    def __init__(self):
        # محاكاة API الأدوية - في التطبيق الحقيقي، استبدل بـ API حقيقي
        self.mock_drug_database = {
            "paracetamol": {
                "name_ar": "باراسيتامول",
                "name_en": "Paracetamol",
                "concentrations": ["500mg", "1000mg"],
                "general_use_ar": "مسكن للألم وخافض للحرارة",
                "general_use_en": "Pain reliever and fever reducer",
                "interactions_ar": ["الكحول", "الوارفارين"],
                "interactions_en": ["Alcohol", "Warfarin"],
                "warnings_ar": ["لا يتجاوز 4 جرام يومياً", "حذار من أمراض الكبد"],
                "warnings_en": ["Do not exceed 4g daily", "Caution with liver disease"],
                "alternatives_ar": ["إيبوبروفين", "أسبرين"],
                "alternatives_en": ["Ibuprofen", "Aspirin"],
                "danger_level": "low"
            },
            "ibuprofen": {
                "name_ar": "إيبوبروفين", 
                "name_en": "Ibuprofen",
                "concentrations": ["200mg", "400mg", "600mg"],
                "general_use_ar": "مسكن ومضاد للالتهاب",
                "general_use_en": "Pain reliever and anti-inflammatory",
                "interactions_ar": ["الأسبرين", "مضادات التجلط", "أدوية الضغط"],
                "interactions_en": ["Aspirin", "Blood thinners", "Blood pressure medications"],
                "warnings_ar": ["تجنب مع قرحة المعدة", "حذار مع أمراض الكلى"],
                "warnings_en": ["Avoid with stomach ulcers", "Caution with kidney disease"],
                "alternatives_ar": ["باراسيتامول", "نابروكسين"],
                "alternatives_en": ["Paracetamol", "Naproxen"],
                "danger_level": "medium"
            },
            "warfarin": {
                "name_ar": "وارفارين",
                "name_en": "Warfarin", 
                "concentrations": ["1mg", "2mg", "5mg"],
                "general_use_ar": "مضاد للتجلط",
                "general_use_en": "Blood thinner",
                "interactions_ar": ["الأسبرين", "باراسيتامول", "المضادات الحيوية"],
                "interactions_en": ["Aspirin", "Paracetamol", "Antibiotics"],
                "warnings_ar": ["يتطلب مراقبة دورية للدم", "خطر النزيف"],
                "warnings_en": ["Requires regular blood monitoring", "Bleeding risk"],
                "alternatives_ar": ["ريفاروكسابان", "دابيجاتران"],
                "alternatives_en": ["Rivaroxaban", "Dabigatran"],
                "danger_level": "high"
            }
        }

    def search_drug(self, drug_name: str, language: str = 'ar') -> Optional[Dict]:
        """البحث عن دواء في قاعدة البيانات"""
        drug_name_clean = drug_name.lower().strip()
        
        for key, drug_info in self.mock_drug_database.items():
            if (drug_name_clean in key.lower() or 
                drug_name_clean in drug_info.get('name_ar', '').lower() or
                drug_name_clean in drug_info.get('name_en', '').lower()):
                return drug_info
        
        return None

    def check_dangerous_interactions(self, current_drugs: List[str], new_drug: str) -> Tuple[bool, List[str]]:
        """فحص التداخلات الخطيرة بين الأدوية"""
        new_drug_info = self.search_drug(new_drug)
        if not new_drug_info:
            return False, []
        
        dangerous_interactions = []
        
        for current_drug in current_drugs:
            current_drug_info = self.search_drug(current_drug)
            if current_drug_info:
                # فحص التداخلات
                interactions = new_drug_info.get('interactions_ar', []) + new_drug_info.get('interactions_en', [])
                if any(interaction.lower() in current_drug.lower() for interaction in interactions):
                    dangerous_interactions.append(current_drug)
        
        return len(dangerous_interactions) > 0, dangerous_interactions

class CaseClassifier:
    def __init__(self):
        self.danger_keywords = {
            'ar': [
                'نزيف', 'نزف', 'دم', 'ضيق نفس', 'صعوبة التنفس', 'تورم', 'انتفاخ',
                'طفح جلدي شديد', 'حمى شديدة', 'قيء شديد', 'إسهال شديد', 'دوخة شديدة',
                'ألم صدر', 'خفقان', 'إغماء', 'تشنجات', 'صداع شديد مفاجئ',
                'ضعف مفاجئ', 'تنميل', 'فقدان الوعي', 'حساسية شديدة'
            ],
            'en': [
                'bleeding', 'blood', 'shortness of breath', 'difficulty breathing', 'swelling',
                'severe rash', 'high fever', 'severe vomiting', 'severe diarrhea', 'severe dizziness',
                'chest pain', 'palpitations', 'fainting', 'seizures', 'sudden severe headache',
                'sudden weakness', 'numbness', 'loss of consciousness', 'severe allergy'
            ]
        }
        
        self.pharmacist_keywords = {
            'ar': [
                'حامل', 'حمل', 'رضاعة', 'مرضع', 'طفل', 'رضيع', 'كبير السن',
                'مرض مزمن', 'سكري', 'ضغط', 'كلى', 'كبد', 'قلب', 'تداخل دوائي',
                'حساسية دوائية', 'عدة أدوية', 'جراحة قريبة'
            ],
            'en': [
                'pregnant', 'pregnancy', 'breastfeeding', 'nursing', 'child', 'infant', 'elderly',
                'chronic disease', 'diabetes', 'blood pressure', 'kidney', 'liver', 'heart',
                'drug interaction', 'drug allergy', 'multiple medications', 'recent surgery'
            ]
        }

    def classify_case(self, symptoms: str, user_data: Dict, language: str) -> Dict:
        """تصنيف الحالة إلى: بسيطة، تحتاج صيدلي، طارئة"""
        symptoms_lower = symptoms.lower()
        
        # فحص الكلمات الخطيرة (طوارئ)
        danger_words = self.danger_keywords.get(language, [])
        emergency_detected = any(word in symptoms_lower for word in danger_words)
        
        # فحص عمر الطفل (أقل من 3 شهور)
        age = user_data.get('age', '')
        if 'شهر' in age or 'month' in age.lower():
            try:
                age_months = int(re.findall(r'\d+', age)[0])
                if age_months < 3:
                    emergency_detected = True
            except:
                pass
        
        if emergency_detected:
            return {
                'classification': 'emergency',
                'message_ar': '⚠️ تم اكتشاف أعراض خطيرة. يرجى التوجه لأقرب مستشفى فوراً أو الاتصال بالطوارئ.',
                'message_en': '⚠️ Dangerous symptoms detected. Please go to the nearest hospital immediately or call emergency services.',
                'action': 'stop_medical_response'
            }
        
        # فحص الحاجة لصيدلي
        pharmacist_words = self.pharmacist_keywords.get(language, [])
        needs_pharmacist = any(word in symptoms_lower for word in pharmacist_words)
        
        # فحص الحالات الخاصة من بيانات المستخدم
        special_conditions = [
            user_data.get('chronic_diseases', ''),
            user_data.get('current_medications', ''),
            user_data.get('allergies', '')
        ]
        
        if any(condition.strip() for condition in special_conditions) or needs_pharmacist:
            return {
                'classification': 'needs_pharmacist',
                'message_ar': 'حالتك تتطلب استشارة صيدلي مختص. سيتم تحويلك للصيدلي.',
                'message_en': 'Your case requires pharmacist consultation. You will be referred to a pharmacist.',
                'action': 'refer_to_pharmacist'
            }
        
        return {
            'classification': 'simple',
            'message_ar': 'يمكنني تقديم معلومات عامة عن حالتك.',
            'message_en': 'I can provide general information about your condition.',
            'action': 'provide_general_info'
        }

class PrescriptionOCR:
    def __init__(self):
        self.reader = easyocr.EasyOCR(['ar', 'en'])
        
    def extract_drug_info(self, image) -> Dict:
        """استخراج اسم الدواء والتركيز من الوصفة الطبية"""
        try:
            # تحويل الصورة إلى array
            img_array = np.array(image)
            
            # قراءة النص من الصورة
            results = self.reader.readtext(img_array)
            
            extracted_text = []
            for (bbox, text, confidence) in results:
                if confidence > 0.5:  # فقط النصوص بثقة عالية
                    extracted_text.append(text)
            
            # البحث عن أسماء الأدوية والتراكيز
            drugs_found = []
            drug_api = DrugAPIHandler()
            
            for text in extracted_text:
                # البحث عن تراكيز (mg, gm, ml)
                concentration_match = re.search(r'(\d+)\s*(mg|gm|ml|gram)', text.lower())
                
                # البحث عن أسماء الأدوية
                drug_info = drug_api.search_drug(text)
                if drug_info:
                    concentration = concentration_match.group() if concentration_match else "غير محدد"
                    drugs_found.append({
                        'name': text,
                        'concentration': concentration,
                        'drug_info': drug_info
                    })
            
            return {
                'success': True,
                'drugs_found': drugs_found,
                'raw_text': extracted_text,
                'message_ar': f'تم استخراج {len(drugs_found)} دواء من الوصفة',
                'message_en': f'Extracted {len(drugs_found)} medications from prescription'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message_ar': 'فشل في قراءة الوصفة الطبية',
                'message_en': 'Failed to read prescription'
            }

class PharmacistPanel:
    @staticmethod
    def create_case_summary(user_data: Dict, symptoms: str, drug_query: str, classification: Dict) -> Dict:
        """إنشاء ملخص الحالة للصيدلي"""
        return {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_info': {
                'age': user_data.get('age', 'غير محدد'),
                'weight': user_data.get('weight', 'غير محدد'),
                'chronic_diseases': user_data.get('chronic_diseases', 'لا يوجد'),
                'current_medications': user_data.get('current_medications', 'لا يوجد'),
                'allergies': user_data.get('allergies', 'لا يوجد')
            },
            'case_details': {
                'symptoms': symptoms,
                'drug_query': drug_query,
                'classification': classification['classification'],
                'urgency_level': classification.get('urgency_level', 'medium')
            },
            'warnings': classification.get('warnings', []),
            'case_id': f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }

class AdvancedMedicalChatbot:
    def __init__(self):
        self.setup_models()
        self.drug_api = DrugAPIHandler()
        self.case_classifier = CaseClassifier()
        self.ocr_processor = PrescriptionOCR()
        self.user_data = {}
        self.setup_intents()

    def setup_models(self):
        """تهيئة نماذج mBERT"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.classifier = pipeline(
                "text-classification",
                model="bert-base-multilingual-cased",
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            st.success("✅ تم تحميل النماذج بنجاح!")
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {str(e)}")

    def setup_intents(self):
        """تحديد أنواع الاستفسارات الطبية"""
        self.intents = {
            "collect_user_info": {
                "ar": ["عمر", "سن", "وزن", "كيلو", "أعراض", "حساسية", "مرض مزمن", "أدوية حالية"],
                "en": ["age", "weight", "symptoms", "allergy", "chronic", "current medications"]
            },
            "drug_inquiry": {
                "ar": ["دواء", "علاج", "حبوب", "جرعة", "تأثير", "بديل"],
                "en": ["medicine", "medication", "drug", "dose", "effect", "alternative"]
            },
            "prescription_reading": {
                "ar": ["وصفة", "روشتة", "قراءة", "صورة الوصفة"],
                "en": ["prescription", "read prescription", "prescription image"]
            }
        }

    def collect_user_information(self) -> bool:
        """جمع معلومات المستخدم الأساسية"""
        st.subheader("معلومات المستخدم | User Information")
        
        with st.form("user_info_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.text_input("العمر | Age", placeholder="مثال: 25 سنة / 25 years")
                weight = st.text_input("الوزن | Weight", placeholder="مثال: 70 كيلو / 70 kg")
                chronic_diseases = st.text_area("الأمراض المزمنة | Chronic Diseases", 
                                               placeholder="مثال: سكري، ضغط / Diabetes, Hypertension")
            
            with col2:
                allergies = st.text_area("الحساسية | Allergies",
                                       placeholder="مثال: البنسلين / Penicillin")
                current_medications = st.text_area("الأدوية الحالية | Current Medications",
                                                 placeholder="مثال: الأسبرين يومياً / Aspirin daily")
            
            symptoms = st.text_area("الأعراض الحالية | Current Symptoms",
                                  placeholder="اذكر الأعراض التي تعاني منها / Describe your symptoms")
            
            submitted = st.form_submit_button("حفظ المعلومات | Save Information")
            
            if submitted:
                self.user_data = {
                    'age': age,
                    'weight': weight,
                    'chronic_diseases': chronic_diseases,
                    'allergies': allergies,
                    'current_medications': current_medications,
                    'symptoms': symptoms,
                    'timestamp': datetime.now()
                }
                st.session_state.user_data = self.user_data
                st.success("✅ تم حفظ معلوماتك بنجاح!")
                return True
        
        return False

    def process_drug_inquiry(self, query: str, language: str) -> str:
        """معالجة استفسار عن دواء معين"""
        # استخراج اسم الدواء من الاستفسار
        drug_name = self.extract_drug_name(query)
        
        if not drug_name:
            if language == 'ar':
                return "لم أتمكن من تحديد اسم الدواء. يرجى ذكر اسم الدواء بوضوح."
            else:
                return "I couldn't identify the drug name. Please mention the drug name clearly."
        
        # البحث عن معلومات الدواء
        drug_info = self.drug_api.search_drug(drug_name, language)
        
        if not drug_info:
            if language == 'ar':
                return f"عذراً، لا توجد معلومات عن الدواء '{drug_name}' في قاعدة البيانات."
            else:
                return f"Sorry, no information found for drug '{drug_name}' in the database."
        
        # تصنيف الحالة
        user_data = st.session_state.get('user_data', {})
        classification = self.case_classifier.classify_case(query, user_data, language)
        
        # إذا كانت حالة طارئة
        if classification['action'] == 'stop_medical_response':
            return classification[f'message_{language}']
        
        # فحص التداخلات الخطيرة
        current_medications = user_data.get('current_medications', '').split(',')
        has_interactions, interactions = self.drug_api.check_dangerous_interactions(
            current_medications, drug_name
        )
        
        # إذا كانت تحتاج صيدلي أو يوجد تداخلات خطيرة
        if classification['action'] == 'refer_to_pharmacist' or (has_interactions and drug_info.get('danger_level') == 'high'):
            # إنشاء ملخص للصيدلي
            case_summary = PharmacistPanel.create_case_summary(
                user_data, user_data.get('symptoms', ''), query, classification
            )
            st.session_state.pharmacist_cases = st.session_state.get('pharmacist_cases', [])
            st.session_state.pharmacist_cases.append(case_summary)
            
            warning_msg = ""
            if has_interactions:
                if language == 'ar':
                    warning_msg = f"⚠️ تحذير: يوجد تداخل دوائي محتمل مع: {', '.join(interactions)}\n"
                else:
                    warning_msg = f"⚠️ Warning: Potential drug interaction with: {', '.join(interactions)}\n"
            
            return warning_msg + classification[f'message_{language}'] + f"\n\nرقم الحالة | Case ID: {case_summary['case_id']}"
        
        # إعطاء معلومات عامة فقط للحالات البسيطة
        if language == 'ar':
            response = f"**معلومات عن {drug_info['name_ar']}:**\n\n"
            response += f"🔹 **الاستخدام العام:** {drug_info['general_use_ar']}\n"
            response += f"🔹 **التراكيز المتوفرة:** {', '.join(drug_info['concentrations'])}\n"
            response += f"🔹 **التداخلات الدوائية:** {', '.join(drug_info['interactions_ar'])}\n"
            response += f"🔹 **التحذيرات الأساسية:** {', '.join(drug_info['warnings_ar'])}\n"
            response += f"🔹 **البدائل المتاحة:** {', '.join(drug_info['alternatives_ar'])}\n\n"
            response += "⚠️ **تنبيه:** هذه معلومات عامة فقط. استشر الصيدلي للجرعات والاستخدام المناسب."
        else:
            response = f"**Information about {drug_info['name_en']}:**\n\n"
            response += f"🔹 **General Use:** {drug_info['general_use_en']}\n"
            response += f"🔹 **Available Concentrations:** {', '.join(drug_info['concentrations'])}\n"
            response += f"🔹 **Drug Interactions:** {', '.join(drug_info['interactions_en'])}\n"
            response += f"🔹 **Basic Warnings:** {', '.join(drug_info['warnings_en'])}\n"
            response += f"🔹 **Alternatives:** {', '.join(drug_info['alternatives_en'])}\n\n"
            response += "⚠️ **Note:** This is general information only. Consult a pharmacist for appropriate dosage and usage."
        
        return response

    def extract_drug_name(self, query: str) -> str:
        """استخراج اسم الدواء من الاستفسار"""
        # قائمة الأدوية المعروفة
        known_drugs = list(self.drug_api.mock_drug_database.keys())
        known_drugs.extend([info['name_ar'] for info in self.drug_api.mock_drug_database.values()])
        known_drugs.extend([info['name_en'] for info in self.drug_api.mock_drug_database.values()])
        
        query_lower = query.lower()
        for drug in known_drugs:
            if drug.lower() in query_lower:
                return drug
        
        return ""

    def detect_language(self, text):
        """كشف لغة النص"""
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if len(arabic_chars) > len(text) * 0.3:
            return 'ar'
        return 'en'

def main():
    try:
        st.set_page_config(
            page_title="البوت الطبي التوعوي المتقدم",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.error(f"خطأ في التهيئة: {str(e)}")

    st.title("💊 البوت الطبي التوعوي المتقدم")
    st.markdown("### Advanced Educational Medical Bot | بوت طبي توعوي متقدم مع API الأدوية")

    # تهيئة البوت
    if 'chatbot' not in st.session_state:
        with st.spinner("جاري تحميل النظام..."):
            try:
                st.session_state.chatbot = AdvancedMedicalChatbot()
            except Exception as e:
                st.error(f"خطأ في تحميل النظام: {str(e)}")

    # تهيئة المحادثة المستمرة
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}

    # الشريط الجانبي
    with st.sidebar:
        st.header("الميزات | Features")
        st.markdown("""
        ✅ **تكامل API الأدوية**
        
        ✅ **تصنيف الحالات الذكي**
        
        ✅ **قراءة الوصفات الطبية**
        
        ✅ **لوحة الصيدلي**
        
        ✅ **المحادثة المستمرة**
        
        ✅ **نظام الأمان والتحذيرات**
        """)

        st.header("رفع الوصفة الطبية")
        uploaded_file = st.file_uploader("ارفع صورة الوصفة...", type=['png', 'jpg', 'jpeg'])

        # عرض لوحة الصيدلي
        if st.button("لوحة الصيدلي | Pharmacist Panel"):
            st.session_state.show_pharmacist_panel = True

    # التحقق من وجود معلومات المستخدم
    chatbot = st.session_state.chatbot
    
    if not st.session_state.user_data:
        st.warning("⚠️ يرجى ملء معلوماتك الأساسية أولاً")
        if chatbot.collect_user_information():
            st.rerun()
        return

    # واجهة المحادثة الرئيسية
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("واجهة المحادثة | Chat Interface")

        # عرض تاريخ المحادثة (محفوظ ومستمر)
        if st.session_state.chat_history:
            st.subheader("المحادثة السابقة | Previous Conversation")
            for i, (user_msg, bot_response, timestamp) in enumerate(st.session_state.chat_history):
                st.markdown(f"**أنت ({timestamp}):** {user_msg}")
                st.markdown(f"**البوت:** {bot_response}")
                st.markdown("---")

        # إدخال الرسالة الجديدة
        user_input = st.text_area("اكتب رسالتك (عربي/إنجليزي):", 
                                 placeholder="مثال: أريد معلومات عن دواء الباراسيتامول", 
                                 key="user_input_area")

        col_send, col_clear = st.columns([1, 1])
        
        with col_send:
            if st.button("إرسال | Send", type="primary"):
                if user_input:
                    process_user_message(user_input, uploaded_file)

        with col_clear:
            if st.button("مسح المحادثة | Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

    with col2:
        st.header("معلومات المستخدم")
        user_data = st.session_state.user_data
        if user_data:
            st.info(f"**العمر:** {user_data.get('age', 'غير محدد')}")
            st.info(f"**الوزن:** {user_data.get('weight', 'غير محدد')}")
            if user_data.get('chronic_diseases'):
                st.warning(f"**أمراض مزمنة:** {user_data['chronic_diseases']}")
            if user_data.get('allergies'):
                st.error(f"**حساسية:** {user_data['allergies']}")

        if st.button("تحديث المعلومات | Update Info"):
            st.session_state.user_data = {}
            st.rerun()

    # لوحة الصيدلي
    if st.session_state.get('show_pharmacist_panel', False):
        display_pharmacist_panel()

    # معالجة الوصفة الطبية
    if uploaded_file:
        st.header("تحليل الوصفة الطبية")
        process_prescription(uploaded_file)

def process_user_message(user_input: str, uploaded_file=None):
    """معالجة رسالة المستخدم"""
    chatbot = st.session_state.chatbot
    language = chatbot.detect_language(user_input)
    
    # معالجة الاستفسار
    response = chatbot.process_drug_inquiry(user_input, language)
    
    # إضافة للمحادثة المحفوظة
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append((user_input, response, timestamp))
    
    st.rerun()

def process_prescription(uploaded_file):
    """معالجة الوصفة الطبية المرفوعة"""
    chatbot = st.session_state.chatbot
    
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="الوصفة الطبية المرفوعة", use_column_width=True)
        
        with st.spinner("جاري قراءة الوصفة..."):
            ocr_result = chatbot.ocr_processor.extract_drug_info(image)
        
        if ocr_result['success']:
            st.success(ocr_result['message_ar'])
            
            if ocr_result['drugs_found']:
                st.subheader("الأدوية المستخرجة من الوصفة:")
                
                for drug in ocr_result['drugs_found']:
                    with st.expander(f"💊 {drug['name']} - {drug['concentration']}"):
                        drug_info = drug['drug_info']
                        st.write(f"**الاستخدام:** {drug_info['general_use_ar']}")
                        st.write(f"**التحذيرات:** {', '.join(drug_info['warnings_ar'])}")
                        st.write(f"**التداخلات:** {', '.join(drug_info['interactions_ar'])}")
                        
                        if drug_info['danger_level'] == 'high':
                            st.error("⚠️ هذا الدواء يتطلب مراقبة طبية دقيقة")
            
            # عرض النص الخام المستخرج
            with st.expander("النص المستخرج من الصورة"):
                st.write(ocr_result['raw_text'])
        
        else:
            st.error(ocr_result['message_ar'])
            
    except Exception as e:
        st.error(f"خطأ في معالجة الوصفة: {str(e)}")

def display_pharmacist_panel():
    """عرض لوحة الصيدلي"""
    st.header("🩺 لوحة الصيدلي | Pharmacist Panel")
    
    pharmacist_cases = st.session_state.get('pharmacist_cases', [])
    
    if not pharmacist_cases:
        st.info("لا توجد حالات محولة للصيدلي حالياً")
        if st.button("إغلاق اللوحة"):
            st.session_state.show_pharmacist_panel = False
            st.rerun()
        return
    
    st.write(f"**عدد الحالات المحولة:** {len(pharmacist_cases)}")
    
    for i, case in enumerate(pharmacist_cases):
        with st.expander(f"حالة رقم {i+1} - {case['case_id']}"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("معلومات المريض")
                st.write(f"**العمر:** {case['user_info']['age']}")
                st.write(f"**الوزن:** {case['user_info']['weight']}")
                st.write(f"**الأمراض المزمنة:** {case['user_info']['chronic_diseases']}")
                st.write(f"**الأدوية الحالية:** {case['user_info']['current_medications']}")
                st.write(f"**الحساسية:** {case['user_info']['allergies']}")
            
            with col2:
                st.subheader("تفاصيل الحالة")
                st.write(f"**الأعراض:** {case['case_details']['symptoms']}")
                st.write(f"**الاستفسار:** {case['case_details']['drug_query']}")
                st.write(f"**تصنيف الحالة:** {case['case_details']['classification']}")
                st.write(f"**الوقت:** {case['timestamp']}")
            
            # مساحة للصيدلي للرد
            pharmacist_response = st.text_area(f"رد الصيدلي لحالة {i+1}:", key=f"pharmacist_{i}")
            
            if st.button(f"حفظ رد الحالة {i+1}", key=f"save_{i}"):
                if pharmacist_response:
                    # حفظ رد الصيدلي
                    st.session_state.pharmacist_cases[i]['pharmacist_response'] = pharmacist_response
                    st.session_state.pharmacist_cases[i]['response_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.success("تم حفظ الرد!")

    if st.button("إغلاق لوحة الصيدلي"):
        st.session_state.show_pharmacist_panel = False
        st.rerun()

if __name__ == "__main__":
    main()

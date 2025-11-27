
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
        # مستوى 3: حالات خطرة - تحويل فوري
        self.emergency_keywords = {
            'ar': [
                'نزيف', 'نزف', 'دم', 'ضيق نفس', 'صعوبة التنفس', 'ألم صدر', 'خفقان',
                'إغماء', 'فقدان الوعي', 'تشنجات', 'تشنج', 'تورم شديد', 'انتفاخ شديد',
                'حساسية شديدة', 'طفح شديد', 'بلع دواء خاطئ', 'جرعة زائدة',
                'صعوبة بلع', 'شلل', 'ضعف مفاجئ', 'تنميل مفاجئ'
            ],
            'en': [
                'bleeding', 'blood', 'shortness of breath', 'difficulty breathing', 'chest pain',
                'palpitations', 'fainting', 'loss of consciousness', 'seizures', 'severe swelling',
                'severe allergy', 'severe rash', 'wrong medication', 'overdose',
                'difficulty swallowing', 'paralysis', 'sudden weakness', 'sudden numbness'
            ]
        }
        
        # مستوى 1: حالات بسيطة - مساعد آمن
        self.simple_symptoms = {
            'ar': [
                'صداع', 'وجع راس', 'راسي يعورني', 'زكام', 'رشح', 'انفلونزا',
                'كحة بسيطة', 'سعال بسيط', 'كحة خفيفة', 'حرارة خفيفة', 'حمى بسيطة',
                'مغص بسيط', 'بطني يعورني', 'وجع معدة', 'حساسية خفيفة', 'حكة بسيطة',
                'التهاب حلق خفيف', 'وجع حلق'
            ],
            'en': [
                'headache', 'mild fever', 'simple cough', 'light cold', 'runny nose',
                'sore throat', 'mild stomach ache', 'light allergy', 'mild itching'
            ]
        }
        
        # مستوى 2: تتطلب بيانات إضافية
        self.needs_info_keywords = {
            'ar': [
                'حامل', 'حمل', 'رضاعة', 'مرضع', 'طفل', 'رضيع', 'كبير السن',
                'مرض مزمن', 'سكري', 'ضغط', 'كلى', 'كبد', 'قلب'
            ],
            'en': [
                'pregnant', 'pregnancy', 'breastfeeding', 'nursing', 'child', 'infant', 'elderly',
                'chronic disease', 'diabetes', 'blood pressure', 'kidney', 'liver', 'heart'
            ]
        }
        
        # إضافة قاموس أسماء الأدوية العامية
        self.drug_synonyms = {
            'بنادول': 'paracetamol',
            'بودي': 'paracetamol', 
            'أدول': 'paracetamol',
            'فيفادول': 'paracetamol',
            'تايلينول': 'paracetamol',
            'سيتال': 'paracetamol',
            'بروفين': 'ibuprofen',
            'أدفيل': 'ibuprofen',
            'نوروفين': 'ibuprofen',
            'بلفين': 'ibuprofen',
            'أقمنتين': 'augmentin',
            'كلاريتين': 'loratadine',
            'زيرتك': 'cetirizine',
            'تلفاست': 'fexofenadine',
            'اسبرين': 'aspirin',
            'أسبرين': 'aspirin'
        }

    def classify_case(self, symptoms: str, user_data: Dict, language: str) -> Dict:
        """تصنيف الحالة إلى 3 مستويات: بسيطة، تحتاج بيانات، طارئة"""
        symptoms_lower = symptoms.lower()
        
        # مستوى 3: فحص الحالات الطارئة أولاً
        emergency_words = self.emergency_keywords.get(language, [])
        emergency_detected = any(word in symptoms_lower for word in emergency_words)
        
        # فحص عمر الطفل (أقل من شهرين)
        age = user_data.get('age', '')
        if 'شهر' in age or 'month' in age.lower():
            try:
                age_months = int(re.findall(r'\d+', age)[0])
                if age_months < 2:  # تغيير من 3 إلى 2 شهور
                    emergency_detected = True
            except:
                pass
        
        if emergency_detected:
            return {
                'classification': 'emergency',
                'level': 3,
                'message_ar': '🚨 حالة طارئة: يرجى التوجه لأقرب مستشفى فوراً أو الاتصال بالطوارئ 997',
                'message_en': '🚨 Emergency: Please go to the nearest hospital immediately or call emergency services',
                'action': 'emergency_referral'
            }
        
        # مستوى 2: فحص الحاجة لبيانات إضافية
        needs_info_words = self.needs_info_keywords.get(language, [])
        needs_additional_info = any(word in symptoms_lower for word in needs_info_words)
        
        # فحص إذا البيانات الأساسية ناقصة
        essential_data_missing = not all([
            user_data.get('age', '').strip(),
            user_data.get('weight', '').strip()
        ])
        
        # فحص الحالات الخاصة
        special_conditions = [
            user_data.get('chronic_diseases', ''),
            user_data.get('current_medications', ''),
            user_data.get('allergies', '')
        ]
        has_special_conditions = any(condition.strip() for condition in special_conditions)
        
        if needs_additional_info or essential_data_missing or has_special_conditions:
            return {
                'classification': 'needs_info',
                'level': 2,
                'message_ar': 'أحتاج بعض المعلومات الإضافية لتقديم المساعدة المناسبة',
                'message_en': 'I need additional information to provide appropriate assistance',
                'action': 'collect_additional_info',
                'missing_info': self.get_missing_info(user_data, symptoms_lower, language)
            }
        
        # مستوى 1: فحص الأعراض البسيطة
        simple_words = self.simple_symptoms.get(language, [])
        is_simple_symptom = any(word in symptoms_lower for word in simple_words)
        
        if is_simple_symptom:
            return {
                'classification': 'simple',
                'level': 1,
                'message_ar': 'يمكنني تقديم معلومات ونصائح آمنة لحالتك',
                'message_en': 'I can provide safe information and advice for your condition',
                'action': 'provide_safe_guidance'
            }
        
        # الحالة الافتراضية - طلب توضيح
        return {
            'classification': 'unclear',
            'level': 0,
            'message_ar': 'يرجى توضيح الأعراض أو الاستفسار بشكل أكثر تفصيلاً',
            'message_en': 'Please clarify your symptoms or question in more detail',
            'action': 'request_clarification'
        }
    
    def get_missing_info(self, user_data: Dict, symptoms: str, language: str) -> List[str]:
        """تحديد المعلومات الناقصة"""
        missing = []
        
        if not user_data.get('age', '').strip():
            missing.append('age' if language == 'en' else 'العمر')
        
        if not user_data.get('weight', '').strip():
            missing.append('weight' if language == 'en' else 'الوزن')
        
        # إذا كان فيه ذكر للحمل أو الرضاعة
        pregnancy_words = ['حامل', 'حمل', 'رضاعة', 'مرضع', 'pregnant', 'pregnancy', 'breastfeeding']
        if any(word in symptoms for word in pregnancy_words):
            missing.append('pregnancy status' if language == 'en' else 'حالة الحمل/الرضاعة')
        
        return missing

class PrescriptionOCR:
    def __init__(self):
        self.reader = easyocr.Reader(['ar', 'en'])
        
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
        """معالجة استفسار المستخدم - أعراض أو أدوية"""
        # تحديد نية المستخدم أولاً
        intent = self.detect_user_intent(query, language)
        
        if intent == 'symptom_inquiry':
            return self.handle_symptom_inquiry(query, language)
        elif intent == 'greeting':
            return self.handle_greeting(query, language)
        elif intent == 'drug_inquiry':
            return self.handle_drug_inquiry(query, language)
        else:
            # في حالة عدم التأكد من النية
            return self.handle_general_inquiry(query, language)
        
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
        """استخراج اسم الدواء من الاستفسار مع دعم الأسماء العامية"""
        query_lower = query.lower()
        
        # البحث في الأسماء العامية أولاً
        for colloquial_name, standard_name in self.case_classifier.drug_synonyms.items():
            if colloquial_name in query_lower:
                return standard_name
        
        # قائمة الأدوية المعروفة في قاعدة البيانات
        known_drugs = list(self.drug_api.mock_drug_database.keys())
        known_drugs.extend([info['name_ar'] for info in self.drug_api.mock_drug_database.values()])
        known_drugs.extend([info['name_en'] for info in self.drug_api.mock_drug_database.values()])
        
        for drug in known_drugs:
            if drug.lower() in query_lower:
                return drug
        
        return ""

    def detect_user_intent(self, query: str, language: str) -> str:
        """تحديد نية المستخدم من النص مع دعم العامية العربية"""
        query_lower = query.lower()
        
        # كلمات التحية
        greeting_keywords = {
            'ar': ['هلا', 'مرحبا', 'السلام عليكم', 'أهلا', 'كيفك', 'صباح الخير', 'مساء الخير'],
            'en': ['hello', 'hi', 'hey', 'good morning', 'good evening', 'greetings']
        }
        
        # كلمات الأعراض المحسّنة مع العامية
        symptom_keywords = {
            'ar': [
                # أعراض كلاسيكية
                'صداع', 'ألم', 'وجع', 'حرارة', 'حمى', 'كحة', 'سعال', 'اسعال', 'سعل',
                'دوخة', 'غثيان', 'قيء', 'إسهال', 'إمساك', 'تعب', 'إرهاق', 'ضيق نفس',
                'مغص', 'التهاب', 'طفح', 'حكة', 'تورم', 'انتفاخ', 'خفقان',
                # عبارات عامية
                'راسي يعورني', 'راسي يوجعني', 'صدري ثقيل', 'ولدي يسعل', 'بنتي تسعل',
                'بطني يعورني', 'بطني يوجعني', 'معدتي تعورني', 'حلقي يعورني', 'ظهري يعورني',
                'عيني تعورني', 'أذني تعورني', 'رجلي تعورني', 'يدي تعورني',
                # عبارات الإحساس
                'أعاني من', 'أشعر بـ', 'عندي', 'فيني', 'حاسس بـ', 'تعبان', 'مريض',
                'ما أقدر', 'صعب علي', 'يؤلمني', 'يوجعني', 'يعورني'
            ],
            'en': ['headache', 'pain', 'fever', 'cough', 'dizziness', 'nausea', 'vomiting',
                   'diarrhea', 'constipation', 'tired', 'fatigue', 'shortness of breath',
                   'inflammation', 'rash', 'swelling', 'palpitations', 'i have', 'i feel',
                   'my head hurts', 'my stomach hurts', 'i am sick', 'feeling unwell']
        }
        
        # كلمات الأدوية والاستفسارات العامة عنها
        drug_keywords = {
            'ar': [
                'دواء', 'علاج', 'حبوب', 'كبسولة', 'شراب', 'جرعة', 'مرهم', 'قطرة',
                'وش فايدة', 'إيش فايدة', 'شو فايدة', 'معلومات عن', 'وش يفيد',
                'للزكام', 'للصداع', 'للحمى', 'للسعال', 'للألم', 'للمغص'
            ],
            'en': ['medicine', 'medication', 'drug', 'pill', 'tablet', 'capsule', 'syrup', 'dose',
                   'what is', 'information about', 'tell me about', 'for headache', 'for fever',
                   'for cold', 'for cough', 'for pain']
        }
        
        # فحص التحية
        if any(word in query_lower for word in greeting_keywords.get(language, [])):
            return 'greeting'
        
        # فحص الأعراض مع العامية
        if any(word in query_lower for word in symptom_keywords.get(language, [])):
            return 'symptom_inquiry'
        
        # فحص الأدوية والاستفسارات العامة
        if any(word in query_lower for word in drug_keywords.get(language, [])):
            return 'drug_inquiry'
        
        # البحث عن أسماء أدوية محددة (مع الأسماء العامية)
        if self.extract_drug_name(query):
            return 'drug_inquiry'
        
        return 'general_inquiry'

    def handle_greeting(self, query: str, language: str) -> str:
        """معالجة التحيات"""
        if language == 'ar':
            return """مرحباً بك! 👋 أنا البوت الطبي التوعوي.

يمكنني مساعدتك في:
🔸 معلومات عن الأدوية (مثل: معلومات عن الباراسيتامول)
🔸 الأعراض الصحية (مثل: أعاني من صداع)
🔸 قراءة الوصفات الطبية
🔸 التحذير من تداخل الأدوية

كيف يمكنني مساعدتك اليوم؟"""
        else:
            return """Hello! 👋 I'm your Medical Educational Bot.

I can help you with:
🔸 Medication information (e.g., information about paracetamol)
🔸 Health symptoms (e.g., I have a headache)
🔸 Reading medical prescriptions
🔸 Drug interaction warnings

How can I help you today?"""

    def handle_symptom_inquiry(self, query: str, language: str) -> str:
        """معالجة استفسارات الأعراض مع النظام المحسّن"""
        user_data = st.session_state.get('user_data', {})
        classification = self.case_classifier.classify_case(query, user_data, language)
        
        # مستوى 3: حالة طارئة
        if classification['action'] == 'emergency_referral':
            return classification[f'message_{language}']
        
        # مستوى 2: تحتاج بيانات إضافية
        if classification['action'] == 'collect_additional_info':
            missing_info = classification.get('missing_info', [])
            if language == 'ar':
                response = f"🔍 **{classification['message_ar']}**\n\n"
                response += "المعلومات المطلوبة:\n"
                for info in missing_info:
                    response += f"• {info}\n"
                response += "\nيرجى تحديث معلوماتك من الشريط الجانبي أو الإجابة هنا."
            else:
                response = f"🔍 **{classification['message_en']}**\n\n"
                response += "Required information:\n"
                for info in missing_info:
                    response += f"• {info}\n"
                response += "\nPlease update your information from the sidebar or answer here."
            return response
        
        # مستوى 1: حالة بسيطة - تقديم مساعدة آمنة
        if classification['action'] == 'provide_safe_guidance':
            if language == 'ar':
                response = "💡 **معلومات ونصائح آمنة لحالتك:**\n\n"
            else:
                response = "💡 **Safe information and advice for your condition:**\n\n"
            
            # تحليل الأعراض مع نصائح محددة
            symptoms_advice = self.analyze_symptoms(query, language)
            response += symptoms_advice
            
            # إضافة جرعة الباراسيتامول للأطفال إذا كانت المعلومات متوفرة
            age = user_data.get('age', '')
            weight = user_data.get('weight', '')
            if age and weight and ('طفل' in query or 'child' in query.lower() or 
                                 any(word in age.lower() for word in ['سنة', 'شهر', 'year', 'month'])):
                pediatric_dose = self.calculate_pediatric_paracetamol_dose(age, weight, language)
                if pediatric_dose:
                    response += f"\n\n{pediatric_dose}"
            
            # نصائح عامة
            if language == 'ar':
                response += "\n\n📋 **نصائح عامة:**\n"
                response += "• راحة كافية وسوائل دافئة\n"
                response += "• مراقبة الأعراض وتطورها\n"
                response += "• مراجعة الصيدلي للجرعات المناسبة\n"
                response += "• التوجه للطبيب إذا لم تتحسن الأعراض\n\n"
                response += "⚠️ **تنبيه:** معلومات توعوية عامة، استشر مختص للحالات المعقدة"
            else:
                response += "\n\n📋 **General advice:**\n"
                response += "• Adequate rest and warm fluids\n"
                response += "• Monitor symptoms and progression\n"
                response += "• Consult pharmacist for appropriate doses\n"
                response += "• See doctor if symptoms don't improve\n\n"
                response += "⚠️ **Note:** General educational information, consult specialist for complex cases"
            
            return response
        
        # حالة غير واضحة
        if language == 'ar':
            return """❓ **يرجى توضيح الأعراض بشكل أكثر تفصيلاً**

أمثلة واضحة:
• "راسي يعورني من ساعتين"
• "ولدي عمره 3 سنوات عنده حرارة"
• "أعاني من كحة وألم في الحلق"

هذا يساعدني في تقديم المساعدة المناسبة."""
        else:
            return """❓ **Please describe your symptoms in more detail**

Clear examples:
• "I have had a headache for 2 hours"
• "My 3-year-old child has fever"
• "I have cough and sore throat"

This helps me provide appropriate assistance."""

    def analyze_symptoms(self, query: str, language: str) -> str:
        """تحليل الأعراض وتقديم نصائح عامة"""
        query_lower = query.lower()
        advice = []
        
        if language == 'ar':
            if any(word in query_lower for word in ['صداع', 'وجع راس']):
                advice.append("🔸 للصداع: الراحة في مكان هادئ ومظلم، كمادات باردة على الجبهة")
            
            if any(word in query_lower for word in ['كحة', 'سعال', 'اسعال']):
                advice.append("🔸 للسعال: شرب السوائل الدافئة، العسل والليمون، تجنب المهيجات")
            
            if any(word in query_lower for word in ['حرارة', 'حمى']):
                advice.append("🔸 للحمى: شرب السوائل، الراحة، كمادات باردة، قياس درجة الحرارة بانتظام")
            
            if any(word in query_lower for word in ['مغص', 'ألم معدة']):
                advice.append("🔸 لألم المعدة: تجنب الأطعمة الحارة، شرب النعناع، الراحة")
            
            if not advice:
                advice.append("🔸 للأعراض العامة: الراحة وشرب السوائل مهم جداً")
        
        else:
            if any(word in query_lower for word in ['headache', 'head pain']):
                advice.append("🔸 For headache: Rest in quiet, dark place, cold compress on forehead")
            
            if any(word in query_lower for word in ['cough', 'coughing']):
                advice.append("🔸 For cough: Drink warm fluids, honey and lemon, avoid irritants")
            
            if any(word in query_lower for word in ['fever', 'temperature']):
                advice.append("🔸 For fever: Drink fluids, rest, cold compress, monitor temperature regularly")
            
            if any(word in query_lower for word in ['stomach', 'abdominal pain']):
                advice.append("🔸 For stomach pain: Avoid spicy foods, drink mint tea, rest")
            
            if not advice:
                advice.append("🔸 For general symptoms: Rest and hydration are very important")
        
        return "\n".join(advice)

    def handle_drug_inquiry(self, query: str, language: str) -> str:
        """معالجة استفسارات الأدوية"""
        # استخراج اسم الدواء من الاستفسار
        drug_name = self.extract_drug_name(query)
        
        if not drug_name:
            if language == 'ar':
                return """لم أتمكن من تحديد اسم الدواء المحدد. 

**الأدوية المتوفرة في قاعدة البيانات:**
• باراسيتامول (Paracetamol)
• إيبوبروفين (Ibuprofen)  
• وارفارين (Warfarin)

يمكنك السؤال عن أي منها، مثل: "معلومات عن الباراسيتامول" """
            else:
                return """I couldn't identify the specific drug name.

**Available drugs in database:**
• Paracetamol
• Ibuprofen
• Warfarin

You can ask about any of them, e.g., "information about paracetamol" """
        
        # باقي الكود كما هو لمعالجة الأدوية المحددة
        return self.process_specific_drug(drug_name, query, language)

    def calculate_pediatric_paracetamol_dose(self, age_str: str, weight_str: str, language: str) -> str:
        """حساب جرعة الباراسيتامول للأطفال"""
        try:
            # استخراج الوزن
            weight_match = re.findall(r'(\d+\.?\d*)', weight_str)
            if not weight_match:
                return ""
            
            weight = float(weight_match[0])
            
            # استخراج العمر
            age_match = re.findall(r'(\d+)', age_str)
            if not age_match:
                return ""
            
            age = int(age_match[0])
            
            # فحص إذا كان طفل أقل من 6 شهور مع إيبوبروفين
            if 'شهر' in age_str or 'month' in age_str.lower():
                age_months = age
                if age_months < 6:
                    if language == 'ar':
                        return "⚠️ الأطفال أقل من 6 شهور: الباراسيتامول فقط، تجنب الإيبوبروفين"
                    else:
                        return "⚠️ Children under 6 months: Paracetamol only, avoid Ibuprofen"
            
            # حساب جرعة الباراسيتامول (10-15 mg/kg)
            min_dose = weight * 10
            max_dose = weight * 15
            
            # تحويل لملليلتر إذا كان شراب (120mg/5ml)
            syrup_min = (min_dose / 120) * 5
            syrup_max = (max_dose / 120) * 5
            
            if language == 'ar':
                return f"""💊 **جرعة الباراسيتامول المقترحة:**
• **الجرعة:** {min_dose:.0f}-{max_dose:.0f} ملجم
• **الشراب:** {syrup_min:.1f}-{syrup_max:.1f} مل (كل 6-8 ساعات)
• **أقصى 4 جرعات يومياً**

⚠️ استشر الصيدلي قبل الإعطاء"""
            else:
                return f"""💊 **Suggested Paracetamol Dose:**
• **Dose:** {min_dose:.0f}-{max_dose:.0f} mg
• **Syrup:** {syrup_min:.1f}-{syrup_max:.1f} ml (every 6-8 hours)
• **Maximum 4 doses daily**

⚠️ Consult pharmacist before administration"""
                
        except:
            return ""

    def handle_general_inquiry(self, query: str, language: str) -> str:
        """معالجة الاستفسارات العامة مع دعم الأسئلة الشائعة"""
        query_lower = query.lower()
        
        # الأسئلة الشائعة عن الأدوية
        common_questions = {
            'ar': {
                'فايدة بندول': 'بنادول (باراسيتامول) مسكن للألم وخافض للحرارة، آمن للأطفال والكبار',
                'دواء للزكام': 'للزكام: راحة، سوائل دافئة، باراسيتامول للألم، مضادات الاحتقان عند الحاجة',
                'دواء للصداع': 'للصداع: باراسيتامول أو إيبوبروفين، راحة في مكان هادئ، كمادات باردة',
                'دواء للحمى': 'للحمى: باراسيتامول أو إيبوبروفين، سوائل كثيرة، كمادات باردة',
                'دواء للسعال': 'للسعال: عسل وليمون، سوائل دافئة، شراب السعال عند الحاجة'
            },
            'en': {
                'panadol benefits': 'Panadol (paracetamol) is a pain reliever and fever reducer, safe for children and adults',
                'medicine for cold': 'For cold: rest, warm fluids, paracetamol for pain, decongestants when needed',
                'medicine for headache': 'For headache: paracetamol or ibuprofen, rest in quiet place, cold compress',
                'medicine for fever': 'For fever: paracetamol or ibuprofen, plenty of fluids, cold compress',
                'medicine for cough': 'For cough: honey and lemon, warm fluids, cough syrup when needed'
            }
        }
        
        # البحث في الأسئلة الشائعة
        for question, answer in common_questions.get(language, {}).items():
            if question in query_lower:
                return f"📋 **معلومات عامة:**\n\n{answer}\n\n⚠️ هذه معلومات توعوية عامة، استشر الصيدلي للجرعات المحددة."
        
        # الرد الافتراضي
        if language == 'ar':
            return """لم أفهم طلبك بوضوح. يمكنني مساعدتك في:

🔸 **الأعراض:** "راسي يعورني" أو "عندي كحة" أو "بطني يعورني"
🔸 **الأدوية:** "وش فايدة بندول؟" أو "دواء للصداع" 
🔸 **الأسماء العامية:** بندول، بروفين، أدول، فيفادول
🔸 **الوصفات:** ارفع صورة الوصفة من الشريط الجانبي

أمثلة: "معلومات عن بندول" أو "ولدي يسعل" أو "دواء للزكام" """
        else:
            return """I didn't understand your request clearly. I can help you with:

🔸 **Symptoms:** "I have a headache" or "my child has fever"
🔸 **Medications:** "What is panadol for?" or "medicine for cold"
🔸 **Brand names:** Panadol, Adol, Fevadol, Profin
🔸 **Prescriptions:** Upload prescription image from sidebar

Examples: "information about panadol" or "medicine for headache" """

    def process_specific_drug(self, drug_name: str, query: str, language: str) -> str:
        """معالجة دواء محدد - الكود الأصلي"""
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

    # تهيئة المحادثة المستمرة
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}

    # تهيئة البوت
    if 'chatbot' not in st.session_state:
        with st.spinner("جاري تحميل النظام..."):
            try:
                st.session_state.chatbot = AdvancedMedicalChatbot()
            except Exception as e:
                st.error(f"خطأ في تحميل النظام: {str(e)}")
                st.stop()

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


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
                "danger_level": "low",
                "pediatric_safe": True,
                "min_age_months": 0
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
                "danger_level": "medium",
                "pediatric_safe": True,
                "min_age_months": 6
            },
            "mefenamic_acid": {
                "name_ar": "حمض الميفيناميك",
                "name_en": "Mefenamic Acid",
                "concentrations": ["250mg", "500mg"],
                "general_use_ar": "مسكن قوي ومضاد للالتهاب",
                "general_use_en": "Strong pain reliever and anti-inflammatory",
                "interactions_ar": ["مضادات التجلط", "أدوية الضغط", "الليثيوم"],
                "interactions_en": ["Blood thinners", "Blood pressure medications", "Lithium"],
                "warnings_ar": ["لا يستخدم للأطفال أقل من 14 سنة", "حذار من القرحة"],
                "warnings_en": ["Not for children under 14", "Caution with ulcers"],
                "alternatives_ar": ["إيبوبروفين", "ديكلوفيناك"],
                "alternatives_en": ["Ibuprofen", "Diclofenac"],
                "danger_level": "medium",
                "pediatric_safe": False,
                "min_age_months": 168  # 14 years
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
                "danger_level": "high",
                "pediatric_safe": False,
                "min_age_months": 216  # 18 years
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

class AdvancedSymptomParser:
    def __init__(self):
        # Task 6: معالجة النص العامي (Slang Normalization)
        self.slang_normalization = {
            'يعورني': 'ألم',
            'يوجعني': 'ألم', 
            'تعورني': 'ألم',
            'توجعني': 'ألم',
            'كحه': 'سعال',
            'كحة': 'سعال',
            'يكح': 'سعال',
            'اسعل': 'سعال',
            'اسعال': 'سعال',
            'يلتهب': 'التهاب',
            'يلوع': 'غثيان',
            'حلقي يلعب': 'التهاب حلق',
            'صدري يسكر': 'ضيق تنفس',
            'رأسي يعورني': 'صداع',
            'راسي يعورني': 'صداع',
            'بطني يعورني': 'ألم معدة',
            'معدتي تعورني': 'ألم معدة'
        }
        
        # Task 1: تحسين نظام فهم الأعراض - الحالات الطارئة
        self.emergency_symptoms = {
            'ar': [
                'ضيق نفس', 'ضيقة نفس', 'صعوبة تنفس', 'صعوبة في التنفس', 'اختناق',
                'تورم الوجه', 'تورم الشفاه', 'تورم في الوجه', 'انتفاخ الوجه',
                'طفح جلدي شديد', 'طفح شديد', 'حساسية شديدة', 'حكة شديدة في كل الجسم',
                'تشنجات', 'تشنج', 'نوبة', 'رجفة شديدة',
                'فقدان القدرة على الكلام', 'لا أستطيع الكلام', 'صعوبة في الكلام',
                'فقدان القدرة على الحركة', 'شلل', 'لا أستطيع الحركة',
                'قيء شديد مستمر', 'استفراغ مستمر', 'تقيؤ لا يتوقف',
                'نزيف شديد', 'دم كثير', 'نزف',
                'ألم صدر شديد', 'ألم في القلب', 'خفقان شديد'
            ],
            'en': [
                'shortness of breath', 'difficulty breathing', 'cant breathe',
                'facial swelling', 'lip swelling', 'face swollen',
                'severe rash', 'severe allergy', 'severe itching all over',
                'seizures', 'convulsions', 'fits',
                'cannot speak', 'difficulty speaking', 'speech problems',
                'cannot move', 'paralysis', 'weakness',
                'severe vomiting', 'continuous vomiting', 'wont stop vomiting',
                'severe bleeding', 'heavy bleeding',
                'severe chest pain', 'heart pain', 'severe palpitations'
            ]
        }
        
        # Task 1: الأعراض العادية
        self.normal_symptoms = {
            'ar': [
                'صداع خفيف', 'صداع بسيط', 'وجع راس خفيف',
                'زكام', 'رشح', 'انف مسدود',
                'كحة خفيفة', 'سعال بسيط', 'كحة يابسة',
                'حرارة خفيفة', 'حمى بسيطة', 'سخونة خفيفة',
                'مغص بسيط', 'ألم معدة خفيف', 'غازات',
                'حكة بسيطة', 'التهاب حلق خفيف'
            ],
            'en': [
                'mild headache', 'light headache',
                'runny nose', 'stuffy nose', 'cold',
                'mild cough', 'dry cough', 'light cough',
                'mild fever', 'low grade fever',
                'mild stomach ache', 'gas pain',
                'mild itching', 'mild sore throat'
            ]
        }
        
        # Task 1: أعراض تحتاج معلومات إضافية
        self.needs_info_symptoms = {
            'ar': [
                'حرارة عالية', 'حمى شديدة', 'سخونة عالية',
                'قيء', 'استفراغ', 'غثيان شديد',
                'إسهال شديد', 'إسهال مستمر',
                'ألم شديد', 'وجع قوي',
                'دوخة شديدة', 'دوار',
                'طفح جلدي', 'حساسية',
                'كحة مستمرة', 'سعال لا يتوقف'
            ],
            'en': [
                'high fever', 'severe fever',
                'vomiting', 'nausea', 'severe nausea',
                'severe diarrhea', 'continuous diarrhea',
                'severe pain', 'intense pain',
                'severe dizziness', 'vertigo',
                'rash', 'skin rash', 'allergy',
                'persistent cough', 'continuous cough'
            ]
        }

        # Task 2: تحسين اكتشاف الأدوية
        self.drug_synonyms = {
            # باراسيتامول
            'فيفادول': 'paracetamol',
            'بندول': 'paracetamol',
            'بنادول': 'paracetamol',
            'أدول': 'paracetamol',
            'تايلينول': 'paracetamol',
            'سيتال': 'paracetamol',
            'سيتامول': 'paracetamol',
            'panadol': 'paracetamol',
            'fevadol': 'paracetamol',
            'adol': 'paracetamol',
            'tylenol': 'paracetamol',
            
            # إيبوبروفين
            'بروفين': 'ibuprofen',
            'أدفيل': 'ibuprofen',
            'نوروفين': 'ibuprofen',
            'بلفين': 'ibuprofen',
            'موترين': 'ibuprofen',
            'profin': 'ibuprofen',
            'advil': 'ibuprofen',
            'nurofen': 'ibuprofen',
            'motrin': 'ibuprofen',
            
            # حمض الميفيناميك
            'بونستان': 'mefenamic_acid',
            'بونستال': 'mefenamic_acid',
            'ponstan': 'mefenamic_acid',
            'ponstal': 'mefenamic_acid',
            
            # أخرى
            'أسبرين': 'aspirin',
            'اسبرين': 'aspirin',
            'aspirin': 'aspirin',
            'وارفارين': 'warfarin',
            'warfarin': 'warfarin'
        }

    def normalize_text(self, text: str) -> str:
        """تطبيع النص العامي إلى فصيح"""
        normalized = text.lower()
        for slang, formal in self.slang_normalization.items():
            normalized = normalized.replace(slang, formal)
        return normalized

    def extract_drug_names(self, text: str) -> List[str]:
        """استخراج أسماء الأدوية من النص"""
        text_lower = text.lower()
        found_drugs = []
        
        for synonym, standard_name in self.drug_synonyms.items():
            if synonym in text_lower:
                found_drugs.append(standard_name)
        
        return list(set(found_drugs))  # إزالة التكرار

    def classify_symptom_urgency(self, text: str, user_data: Dict, language: str) -> Dict:
        """Task 1: تصنيف الأعراض لثلاث مستويات"""
        normalized_text = self.normalize_text(text)
        
        # Task 5: فحص قواعد الأطفال أولاً
        age = user_data.get('age', '')
        if age:
            age_check = self.check_pediatric_rules(age, normalized_text, language)
            if age_check['action'] == 'emergency_referral':
                return age_check
        
        # فحص الحالات الطارئة
        emergency_words = self.emergency_symptoms.get(language, [])
        if any(word in normalized_text for word in emergency_words):
            return {
                'level': 3,
                'classification': 'emergency',
                'action': 'emergency_referral',
                'message_ar': '🚨 حالة طارئة: توجه للمستشفى فوراً أو اتصل بالطوارئ 997',
                'message_en': '🚨 Emergency: Go to hospital immediately or call emergency 997'
            }
        
        # فحص الأعراض التي تحتاج معلومات إضافية
        needs_info_words = self.needs_info_symptoms.get(language, [])
        if any(word in normalized_text for word in needs_info_words):
            # فحص إذا كانت المعلومات الأساسية ناقصة
            missing_info = self.get_missing_essential_info(user_data, language)
            if missing_info:
                return {
                    'level': 2,
                    'classification': 'needs_info',
                    'action': 'ask_one_question',
                    'message_ar': f'أحتاج معلومة واحدة: {missing_info[0]}',
                    'message_en': f'I need one piece of information: {missing_info[0]}',
                    'missing_info': missing_info[0]
                }
        
        # فحص الأعراض العادية
        normal_words = self.normal_symptoms.get(language, [])
        if any(word in normalized_text for word in normal_words):
            return {
                'level': 1,
                'classification': 'normal',
                'action': 'provide_simple_advice',
                'message_ar': 'يمكنني تقديم نصيحة بسيطة ودواء مناسب',
                'message_en': 'I can provide simple advice and suggest appropriate medication'
            }
        
        # حالة غير واضحة
        return {
            'level': 0,
            'classification': 'unclear',
            'action': 'ask_one_question',
            'message_ar': 'وضح الأعراض أكثر، مثل: راسي يعورني من ساعتين',
            'message_en': 'Clarify symptoms more, like: I have had a headache for 2 hours',
            'missing_info': 'symptom_details'
        }

    def check_pediatric_rules(self, age_str: str, text: str, language: str) -> Dict:
        """Task 5: فحص قواعد الأطفال"""
        try:
            # استخراج العمر بالشهور
            if 'شهر' in age_str or 'month' in age_str.lower():
                age_match = re.findall(r'(\d+)', age_str)
                if age_match:
                    age_months = int(age_match[0])
                    
                    # أقل من 3 شهور + حرارة = طوارئ
                    if age_months < 3 and ('حرارة' in text or 'حمى' in text or 'fever' in text):
                        return {
                            'level': 3,
                            'classification': 'emergency',
                            'action': 'emergency_referral',
                            'message_ar': '🚨 طفل أقل من 3 شهور مع حرارة - توجه للمستشفى فوراً',
                            'message_en': '🚨 Child under 3 months with fever - go to hospital immediately'
                        }
                    
                    # أقل من سنتين = تحويل إجباري
                    if age_months < 24:
                        return {
                            'level': 2,
                            'classification': 'needs_pharmacist',
                            'action': 'refer_to_pharmacist',
                            'message_ar': 'الأطفال أقل من سنتين يحتاجون استشارة صيدلي مختص',
                            'message_en': 'Children under 2 years need specialist pharmacist consultation'
                        }
            
            # فحص العمر بالسنوات
            elif 'سنة' in age_str or 'year' in age_str.lower():
                age_match = re.findall(r'(\d+)', age_str)
                if age_match:
                    age_years = int(age_match[0])
                    if age_years < 2:
                        return {
                            'level': 2,
                            'classification': 'needs_pharmacist',
                            'action': 'refer_to_pharmacist',
                            'message_ar': 'الأطفال أقل من سنتين يحتاجون استشارة صيدلي مختص',
                            'message_en': 'Children under 2 years need specialist pharmacist consultation'
                        }
        except:
            pass
        
        return {'action': 'continue'}  # لا توجد قيود

    def get_missing_essential_info(self, user_data: Dict, language: str) -> List[str]:
        """الحصول على المعلومات الأساسية الناقصة - سؤال واحد فقط"""
        missing = []
        
        if not user_data.get('age', '').strip():
            missing.append('العمر' if language == 'ar' else 'age')
        elif not user_data.get('weight', '').strip():
            missing.append('الوزن' if language == 'ar' else 'weight')
        
        return missing[:1]  # سؤال واحد فقط

class CaseClassifier:
    def __init__(self):
        self.symptom_parser = AdvancedSymptomParser()
        self.drug_api = DrugAPIHandler()

    def classify_case(self, user_input: str, user_data: Dict, language: str) -> Dict:
        """Task 7: Flow واضح للسؤال (Decision Tree)"""
        
        # Step 1: Check for Emergency Symptoms
        symptom_classification = self.symptom_parser.classify_symptom_urgency(user_input, user_data, language)
        if symptom_classification['action'] == 'emergency_referral':
            return symptom_classification
        
        # Step 2: Check for Child Age/Weight Rules
        if symptom_classification['action'] == 'refer_to_pharmacist':
            return symptom_classification
        
        # Step 3: Drug Detected?
        detected_drugs = self.symptom_parser.extract_drug_names(user_input)
        if detected_drugs:
            return {
                'classification': 'drug_inquiry',
                'action': 'provide_drug_info',
                'detected_drugs': detected_drugs,
                'message_ar': 'تم اكتشاف استفسار عن دواء',
                'message_en': 'Drug inquiry detected'
            }
        
        # Step 4: Symptom Detected?
        if symptom_classification['level'] >= 1:
            return symptom_classification
        
        # Step 5: Unknown - Ask one question only
        if symptom_classification['action'] == 'ask_one_question':
            return symptom_classification
        
        # Default case
        return {
            'classification': 'unclear',
            'action': 'request_clarification',
            'message_ar': 'لم أفهم طلبك، هل تسأل عن دواء أم عرض؟',
            'message_en': 'I didnt understand, are you asking about a drug or symptom?'
        }

class AdvancedMedicalChatbot:
    def __init__(self):
        self.setup_models()
        self.drug_api = DrugAPIHandler()
        self.case_classifier = CaseClassifier()
        self.user_data = {}

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

    def detect_user_intent(self, query: str, language: str) -> str:
        """Task 3: فصل بين سؤال "عرض" و"دواء" """
        query_lower = query.lower()
        
        # فحص وجود أسماء أدوية محددة
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(query)
        has_drug_names = len(detected_drugs) > 0
        
        # فحص وجود أعراض
        normalized_text = self.case_classifier.symptom_parser.normalize_text(query)
        symptom_words = self.case_classifier.symptom_parser.normal_symptoms.get(language, [])
        symptom_words += self.case_classifier.symptom_parser.needs_info_symptoms.get(language, [])
        symptom_words += self.case_classifier.symptom_parser.emergency_symptoms.get(language, [])
        has_symptoms = any(word in normalized_text for word in symptom_words)
        
        # فحص الأسئلة العامة عن الأدوية
        drug_general_questions = {
            'ar': ['دواء للصداع', 'دواء للحمى', 'دواء للسعال', 'علاج للزكام', 'وش فايدة', 'معلومات عن'],
            'en': ['medicine for headache', 'medicine for fever', 'drug for', 'what is', 'information about']
        }
        has_drug_question = any(phrase in query_lower for phrase in drug_general_questions.get(language, []))
        
        # Task 3: المنطق الواضح
        if has_symptoms and not has_drug_names:
            return 'symptom_only'
        elif has_drug_names and not has_symptoms:
            return 'drug_only'
        elif has_symptoms and has_drug_names:
            # الأولوية للتحويل لو حالة خطيرة
            return 'mixed_priority_safety'
        elif has_drug_question:
            return 'drug_general_question'
        else:
            return 'unclear'

    def process_query(self, user_input: str, language: str) -> str:
        """معالجة الاستفسار الرئيسية"""
        user_data = st.session_state.get('user_data', {})
        
        # تحديد نوع الاستفسار
        intent = self.detect_user_intent(user_input, language)
        
        if intent == 'symptom_only':
            return self.handle_symptom_inquiry(user_input, user_data, language)
        elif intent == 'drug_only':
            return self.handle_drug_inquiry(user_input, language)
        elif intent == 'mixed_priority_safety':
            return self.handle_mixed_inquiry(user_input, user_data, language)
        elif intent == 'drug_general_question':
            return self.handle_general_drug_question(user_input, language)
        else:
            return self.handle_unclear_query(user_input, language)

    def handle_symptom_inquiry(self, user_input: str, user_data: Dict, language: str) -> str:
        """معالجة استفسارات الأعراض فقط"""
        classification = self.case_classifier.classify_case(user_input, user_data, language)
        
        if classification['action'] == 'emergency_referral':
            return classification[f'message_{language}']
        
        elif classification['action'] == 'refer_to_pharmacist':
            return self.create_pharmacist_referral(classification, user_input, user_data, language)
        
        elif classification['action'] == 'ask_one_question':
            return classification[f'message_{language}']
        
        elif classification['action'] == 'provide_simple_advice':
            return self.provide_symptom_advice_with_drug(user_input, user_data, language)
        
        return classification[f'message_{language}']

    def handle_drug_inquiry(self, user_input: str, language: str) -> str:
        """معالجة استفسارات الأدوية فقط"""
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(user_input)
        
        if not detected_drugs:
            if language == 'ar':
                return "لم أتعرف على اسم الدواء. جرب: فيفادول، بروفين، بندول، أدول"
            else:
                return "Drug name not recognized. Try: Panadol, Profin, Fevadol, Adol"
        
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)
        
        if not drug_info:
            if language == 'ar':
                return f"معلومات الدواء '{drug_name}' غير متوفرة في قاعدة البيانات"
            else:
                return f"Drug information for '{drug_name}' not available in database"
        
        # إعطاء معلومات الدواء
        if language == 'ar':
            response = f"💊 **{drug_info['name_ar']} ({drug_info['name_en']})**\n\n"
            response += f"🔹 **الاستخدام:** {drug_info['general_use_ar']}\n"
            response += f"🔹 **التراكيز:** {', '.join(drug_info['concentrations'])}\n"
            response += f"🔹 **تحذيرات مهمة:** {', '.join(drug_info['warnings_ar'][:2])}\n\n"
            
            # Task 4: بدائل الدواء مبسطة
            if drug_info['alternatives_ar']:
                response += f"**🔄 ما لقيت {drug_info['name_ar']}؟**\n"
                response += f"خذ أي شيء من نفس المجموعة: {', '.join(drug_info['alternatives_ar'][:2])}\n\n"
            
            response += "⚠️ استشر الصيدلي للجرعة المناسبة"
        else:
            response = f"💊 **{drug_info['name_en']} ({drug_info['name_ar']})**\n\n"
            response += f"🔹 **Use:** {drug_info['general_use_en']}\n"
            response += f"🔹 **Strengths:** {', '.join(drug_info['concentrations'])}\n"
            response += f"🔹 **Important warnings:** {', '.join(drug_info['warnings_en'][:2])}\n\n"
            
            if drug_info['alternatives_en']:
                response += f"**🔄 Can't find {drug_info['name_en']}?**\n"
                response += f"Try alternatives: {', '.join(drug_info['alternatives_en'][:2])}\n\n"
            
            response += "⚠️ Consult pharmacist for appropriate dose"
        
        return response

    def handle_mixed_inquiry(self, user_input: str, user_data: Dict, language: str) -> str:
        """معالجة الاستفسارات المختلطة مع الأولوية للأمان"""
        # فحص الأعراض أولاً للأمان
        classification = self.case_classifier.classify_case(user_input, user_data, language)
        
        if classification['action'] == 'emergency_referral':
            return classification[f'message_{language}']
        
        # إذا كانت آمنة، اعرض معلومات الدواء مع تحذيرات
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(user_input)
        drug_response = self.handle_drug_inquiry(user_input, language)
        
        # إضافة تحذير للأعراض
        symptom_warning = ""
        if language == 'ar':
            symptom_warning = "\n\n⚠️ **تنبيه:** لاحظت أنك تذكر أعراض. إذا كانت شديدة أو مستمرة، راجع طبيب."
        else:
            symptom_warning = "\n\n⚠️ **Notice:** I noticed you mentioned symptoms. If severe or persistent, see a doctor."
        
        return drug_response + symptom_warning

    def handle_general_drug_question(self, user_input: str, language: str) -> str:
        """Task 4: معالجة الأسئلة العامة عن الأدوية بشكل مبسط"""
        query_lower = user_input.lower()
        
        # أسئلة شائعة مبسطة
        if language == 'ar':
            if any(word in query_lower for word in ['دواء للصداع', 'علاج للصداع']):
                return """💊 **للصداع العادي:**
• **بندول** (باراسيتامول) - آمن ومجرب
• **بروفين** (إيبوبروفين) - قوي أكثر

**الجرعة:** حسب العمر والوزن
**نصيحة:** راحة + ماء كثير

⚠️ استشر الصيدلي للجرعة المحددة"""

            elif any(word in query_lower for word in ['دواء للحمى', 'دواء للحرارة']):
                return """🌡️ **لخفض الحرارة:**
• **فيفادول** أو **أدول** - للأطفال والكبار
• **بروفين** - قوي ومضاد للالتهاب

**مهم:** كمادات باردة + سوائل كثيرة

⚠️ الأطفال أقل من 6 شهور: بندول فقط"""

            elif any(word in query_lower for word in ['دواء للسعال', 'دواء للكحة']):
                return """🫁 **للسعال:**
• **عسل + ليمون** - طبيعي وفعال
• **شراب السعال** - للكحة الناشفة
• **بروفين** - إذا فيه التهاب

**نصيحة:** سوائل دافئة مهمة جداً

⚠️ السعال أكثر من أسبوع = راجع طبيب"""

        else:  # English
            if 'headache' in query_lower:
                return """💊 **For headache:**
• **Panadol** (Paracetamol) - safe and proven
• **Profin** (Ibuprofen) - stronger

**Dose:** according to age and weight
**Tip:** rest + plenty of water

⚠️ Consult pharmacist for specific dose"""

        # رد افتراضي للأسئلة العامة
        if language == 'ar':
            return """أسئلة شائعة يمكنني الإجابة عليها:

🔹 "دواء للصداع"
🔹 "دواء للحمى" 
🔹 "دواء للسعال"
🔹 "معلومات عن بندول"

أو اسأل عن دواء محدد مثل: فيفادول، بروفين، أدول"""
        else:
            return """Common questions I can answer:

🔹 "medicine for headache"
🔹 "medicine for fever"
🔹 "medicine for cough" 
🔹 "information about panadol"

Or ask about specific drugs like: Panadol, Profin, Adol"""

    def provide_symptom_advice_with_drug(self, user_input: str, user_data: Dict, language: str) -> str:
        """Task 1: تقديم نصيحة بسيطة + دواء مناسب"""
        normalized_text = self.case_classifier.symptom_parser.normalize_text(user_input)
        advice = ""
        
        if language == 'ar':
            if 'صداع' in normalized_text:
                advice = """💡 **للصداع الخفيف:**

**العلاج:**
• بندول 500mg أو فيفادول (كل 6 ساعات)
• أو بروفين 200mg (كل 8 ساعات)

**نصائح:**
• راحة في مكان هادئ
• كمادة باردة على الجبهة
• شرب ماء كافي"""

            elif 'سعال' in normalized_text or 'كحة' in normalized_text:
                advice = """🫁 **للسعال البسيط:**

**العلاج:**
• عسل + ليمون (طبيعي ومفيد)
• شراب السعال (للكحة الناشفة)
• بروفين إذا فيه ألم في الحلق

**نصائح:**
• سوائل دافئة كثيرة
• تجنب المهيجات"""

            elif 'حرارة' in normalized_text or 'حمى' in normalized_text:
                advice = """🌡️ **للحمى البسيطة:**

**العلاج:**
• باراسيتامول (فيفادول/بندول) كل 6 ساعات
• إيبوبروفين (بروفين) كل 8 ساعات

**نصائح:**
• كمادات باردة
• سوائل كثيرة
• راحة تامة"""

            elif 'ألم معدة' in normalized_text or 'مغص' in normalized_text:
                advice = """🫄 **لألم المعدة البسيط:**

**العلاج:**
• تجنب الأطعمة الحارة
• شاي النعناع مفيد
• بندول إذا فيه ألم

**نصائح:**
• أكل خفيف
• سوائل دافئة
• راحة"""

            # Task 5: إضافة جرعة الأطفال إذا توفرت المعلومات
            age = user_data.get('age', '')
            weight = user_data.get('weight', '')
            if age and weight:
                pediatric_info = self.calculate_pediatric_dose(age, weight, language)
                if pediatric_info:
                    advice += f"\n\n{pediatric_info}"

            if not advice:
                advice = "💡 **نصائح عامة للأعراض:**\n• راحة كافية\n• شرب سوائل\n• مراقبة الأعراض"

            advice += "\n\n⚠️ **متى تراجع طبيب:** إذا لم تتحسن خلال 3 أيام أو ازدادت سوءاً"

        else:  # English
            advice = "💡 **General symptom advice:**\n• Adequate rest\n• Drink fluids\n• Monitor symptoms"
            advice += "\n\n⚠️ **See doctor when:** No improvement in 3 days or getting worse"

        return advice

    def calculate_pediatric_dose(self, age_str: str, weight_str: str, language: str) -> str:
        """Task 5: حساب جرعة الأطفال مع قواعد الأمان"""
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
            age_num = int(age_match[0])
            
            # تحديد العمر بالشهور
            if 'شهر' in age_str or 'month' in age_str.lower():
                age_months = age_num
            else:  # سنوات
                age_months = age_num * 12
            
            # Task 5: تطبيق القواعد
            if age_months < 3:
                if language == 'ar':
                    return "⚠️ **أقل من 3 شهور:** تحويل للطبيب - لا أدوية بدون استشارة"
                else:
                    return "⚠️ **Under 3 months:** Refer to doctor - no medicines without consultation"
            
            # حساب الجرعة للباراسيتامول (10-15 mg/kg)
            para_min = weight * 10
            para_max = weight * 15
            
            # الإيبوبروفين (5-10 mg/kg) - فقط أكبر من 6 شهور
            ibuprofen_info = ""
            if age_months >= 6:
                ibu_min = weight * 5
                ibu_max = weight * 10
                if language == 'ar':
                    ibuprofen_info = f"• **بروفين:** {ibu_min:.0f}-{ibu_max:.0f} ملجم (كل 8 ساعات)\n"
                else:
                    ibuprofen_info = f"• **Ibuprofen:** {ibu_min:.0f}-{ibu_max:.0f} mg (every 8 hours)\n"
            
            if language == 'ar':
                result = f"💊 **جرعات الطفل ({age_str}, {weight} كيلو):**\n"
                result += f"• **بندول/فيفادول:** {para_min:.0f}-{para_max:.0f} ملجم (كل 6 ساعات)\n"
                result += ibuprofen_info
                if age_months < 6:
                    result += "⚠️ تحت 6 شهور: بندول فقط، لا بروفين\n"
                result += "**مهم:** أقصى 4 جرعات يومياً"
            else:
                result = f"💊 **Child doses ({age_str}, {weight} kg):**\n"
                result += f"• **Paracetamol:** {para_min:.0f}-{para_max:.0f} mg (every 6 hours)\n"
                result += ibuprofen_info
                if age_months < 6:
                    result += "⚠️ Under 6 months: Paracetamol only, no ibuprofen\n"
                result += "**Important:** Maximum 4 doses daily"
            
            return result
            
        except Exception:
            return ""

    def handle_unclear_query(self, user_input: str, language: str) -> str:
        """Task 9: ردود ديناميكية بدلاً من الثابتة"""
        query_lower = user_input.lower()
        
        # تخمين ذكي بناءً على كلمات مفتاحية
        if language == 'ar':
            if any(word in query_lower for word in ['مرحبا', 'هلا', 'السلام']):
                return """👋 أهلاً وسهلاً!

أنا هنا لمساعدتك في:
🔸 أسئلة عن الأدوية (مثل: وش فايدة بندول؟)
🔸 نصائح للأعراض (مثل: راسي يعورني)
🔸 جرعات الأطفال

وش تحتاج تعرف؟"""

            elif any(word in query_lower for word in ['شكراً', 'شكرا', 'يعطيك العافية']):
                return "العفو! أي سؤال آخر، أنا هنا 😊"
            
            else:
                return f"""لم أفهم '{user_input}' بوضوح.

**جرب هذه الطرق:**
• "راسي يعورني من ساعتين"  
• "معلومات عن بندول"
• "دواء للحمى"
• "ولدي عمره 3 سنوات عنده سعال"

إيش المطلوب بالضبط؟"""
        
        else:  # English
            if any(word in query_lower for word in ['hello', 'hi', 'hey']):
                return """👋 Hello!

I can help you with:
🔸 Drug questions (e.g., what is Panadol for?)
🔸 Symptom advice (e.g., I have a headache)
🔸 Children's doses

What do you need to know?"""
            
            elif any(word in query_lower for word in ['thank', 'thanks']):
                return "You're welcome! Any other questions? 😊"
            
            else:
                return f"""I didn't understand '{user_input}' clearly.

**Try these formats:**
• "I have had a headache for 2 hours"
• "Information about Panadol"  
• "Medicine for fever"
• "My 3-year-old has a cough"

What exactly do you need?"""

    def create_pharmacist_referral(self, classification: Dict, user_input: str, user_data: Dict, language: str) -> str:
        """Task 8: إنشاء إحالة للصيدلي"""
        case_summary = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'user_input': user_input,
            'user_data': user_data,
            'classification': classification,
            'case_id': f"CASE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        }
        
        # إضافة للحالات المحولة
        if 'pharmacist_cases' not in st.session_state:
            st.session_state.pharmacist_cases = []
        st.session_state.pharmacist_cases.append(case_summary)
        
        if language == 'ar':
            return f"""📋 **تم تحويل استفسارك للصيدلي المختص**

**السبب:** {classification['message_ar']}

**رقم الحالة:** {case_summary['case_id']}

سيرد الصيدلي خلال دقائق. يمكنك متابعة الرد من "لوحة الصيدلي" في الشريط الجانبي."""
        else:
            return f"""📋 **Your inquiry has been referred to a specialist pharmacist**

**Reason:** {classification['message_en']}

**Case ID:** {case_summary['case_id']}

The pharmacist will respond within minutes. You can follow up from "Pharmacist Panel" in the sidebar."""

    def detect_language(self, text: str) -> str:
        """كشف لغة النص"""
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        if len(arabic_chars) > len(text) * 0.3:
            return 'ar'
        return 'en'

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

    st.title("💊 البوت الطبي التوعوي المحسّن")
    st.markdown("### Enhanced Educational Medical Bot | بوت طبي توعوي محسّن مع نظام ذكي لفهم الأعراض")

    # تهيئة المحادثة المستمرة
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}

    # تهيئة البوت
    if 'chatbot' not in st.session_state:
        with st.spinner("جاري تحميل النظام المحسّن..."):
            try:
                st.session_state.chatbot = AdvancedMedicalChatbot()
            except Exception as e:
                st.error(f"خطأ في تحميل النظام: {str(e)}")
                st.stop()

    # الشريط الجانبي
    with st.sidebar:
        st.header("الميزات الجديدة | New Features")
        st.markdown("""
        ✅ **نظام ذكي لفهم الأعراض**
        
        ✅ **تصنيف ثلاثي المستويات**
        
        ✅ **معالجة النص العامي**
        
        ✅ **اكتشاف أسماء الأدوية المحسّن**
        
        ✅ **جرعات الأطفال الآمنة**
        
        ✅ **ردود ديناميكية ذكية**
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
        if collect_user_information():
            st.rerun()
        return

    # واجهة المحادثة الرئيسية
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("واجهة المحادثة الذكية | Smart Chat Interface")

        # عرض تاريخ المحادثة
        if st.session_state.chat_history:
            st.subheader("المحادثة | Conversation")
            for i, (user_msg, bot_response, timestamp) in enumerate(st.session_state.chat_history):
                with st.container():
                    st.markdown(f"**أنت ({timestamp}):** {user_msg}")
                    st.markdown(f"**البوت:** {bot_response}")
                    if i < len(st.session_state.chat_history) - 1:
                        st.markdown("---")

        # إدخال الرسالة الجديدة
        user_input = st.text_area("اكتب رسالتك (عربي/إنجليزي):", 
                                 placeholder="مثال: راسي يعورني، أو معلومات عن بندول، أو دواء للحمى", 
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

        # أمثلة للمساعدة
        st.header("أمثلة للتجربة")
        examples = [
            "راسي يعورني من ساعتين",
            "ولدي عمره سنتين عنده حرارة", 
            "معلومات عن بندول",
            "دواء للسعال",
            "بنتي تاخذ فيفادول، آمن؟"
        ]
        
        for example in examples:
            if st.button(f"جرب: {example}", key=f"example_{hash(example)}"):
                st.session_state.user_input_area = example
                st.rerun()

    # لوحة الصيدلي
    if st.session_state.get('show_pharmacist_panel', False):
        display_pharmacist_panel()

    # معالجة الوصفة الطبية
    if uploaded_file:
        st.header("تحليل الوصفة الطبية")
        process_prescription(uploaded_file)

def collect_user_information() -> bool:
    """جمع معلومات المستخدم الأساسية"""
    st.subheader("معلومات المستخدم | User Information")
    
    with st.form("user_info_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.text_input("العمر | Age", placeholder="مثال: 25 سنة / 3 شهور / 25 years")
            weight = st.text_input("الوزن | Weight", placeholder="مثال: 70 كيلو / 12 كيلو / 70 kg")
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
            st.session_state.user_data = {
                'age': age,
                'weight': weight,
                'chronic_diseases': chronic_diseases,
                'allergies': allergies,
                'current_medications': current_medications,
                'symptoms': symptoms,
                'timestamp': datetime.now()
            }
            st.success("✅ تم حفظ معلوماتك بنجاح!")
            return True
    
    return False

def process_user_message(user_input: str, uploaded_file=None):
    """معالجة رسالة المستخدم"""
    chatbot = st.session_state.chatbot
    language = chatbot.detect_language(user_input)
    
    # معالجة الاستفسار بالنظام المحسّن
    response = chatbot.process_query(user_input, language)
    
    # إضافة للمحادثة المحفوظة
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append((user_input, response, timestamp))
    
    st.rerun()

def process_prescription(uploaded_file):
    """معالجة الوصفة الطبية المرفوعة"""
    chatbot = st.session_state.chatbot
    ocr_processor = PrescriptionOCR()
    
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="الوصفة الطبية المرفوعة", use_column_width=True)
        
        with st.spinner("جاري قراءة الوصفة..."):
            ocr_result = ocr_processor.extract_drug_info(image)
        
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
                user_data = case.get('user_data', {})
                st.write(f"**العمر:** {user_data.get('age', 'غير محدد')}")
                st.write(f"**الوزن:** {user_data.get('weight', 'غير محدد')}")
                st.write(f"**الأمراض المزمنة:** {user_data.get('chronic_diseases', 'لا يوجد')}")
                st.write(f"**الأدوية الحالية:** {user_data.get('current_medications', 'لا يوجد')}")
                st.write(f"**الحساسية:** {user_data.get('allergies', 'لا يوجد')}")
            
            with col2:
                st.subheader("تفاصيل الحالة")
                st.write(f"**الاستفسار:** {case.get('user_input', '')}")
                st.write(f"**التصنيف:** {case.get('classification', {}).get('classification', '')}")
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

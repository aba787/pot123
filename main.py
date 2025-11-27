
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
        # قاعدة بيانات شاملة للأدوية الشائعة
        self.mock_drug_database = {
            "paracetamol": {
                "name_ar": "باراسيتامول",
                "name_en": "Paracetamol",
                "concentrations": ["500mg", "1000mg", "120mg/5ml"],
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
                "concentrations": ["200mg", "400mg", "600mg", "100mg/5ml"],
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
            "cetirizine": {
                "name_ar": "سيتيريزين",
                "name_en": "Cetirizine",
                "concentrations": ["10mg", "5mg/5ml"],
                "general_use_ar": "مضاد للحساسية",
                "general_use_en": "Antihistamine for allergies",
                "interactions_ar": ["الكحول", "المهدئات"],
                "interactions_en": ["Alcohol", "Sedatives"],
                "warnings_ar": ["قد يسبب نعاس", "تجنب القيادة"],
                "warnings_en": ["May cause drowsiness", "Avoid driving"],
                "alternatives_ar": ["لوراتادين", "فيكسوفينادين"],
                "alternatives_en": ["Loratadine", "Fexofenadine"],
                "danger_level": "low",
                "pediatric_safe": True,
                "min_age_months": 6
            },
            "loratadine": {
                "name_ar": "لوراتادين",
                "name_en": "Loratadine",
                "concentrations": ["10mg", "5mg/5ml"],
                "general_use_ar": "مضاد للحساسية غير منوم",
                "general_use_en": "Non-drowsy antihistamine",
                "interactions_ar": ["قليلة التداخل"],
                "interactions_en": ["Few interactions"],
                "warnings_ar": ["آمن للاستخدام اليومي"],
                "warnings_en": ["Safe for daily use"],
                "alternatives_ar": ["سيتيريزين", "فيكسوفينادين"],
                "alternatives_en": ["Cetirizine", "Fexofenadine"],
                "danger_level": "low",
                "pediatric_safe": True,
                "min_age_months": 24
            },
            "dextromethorphan": {
                "name_ar": "ديكستروميثورفان",
                "name_en": "Dextromethorphan",
                "concentrations": ["15mg/5ml", "30mg"],
                "general_use_ar": "مضاد للسعال الجاف",
                "general_use_en": "Dry cough suppressant",
                "interactions_ar": ["مضادات الاكتئاب", "MAO inhibitors"],
                "interactions_en": ["Antidepressants", "MAO inhibitors"],
                "warnings_ar": ["لا يستخدم مع السعال المصحوب ببلغم"],
                "warnings_en": ["Not for productive cough"],
                "alternatives_ar": ["العسل", "أدوية طبيعية"],
                "alternatives_en": ["Honey", "Natural remedies"],
                "danger_level": "low",
                "pediatric_safe": True,
                "min_age_months": 24
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
        # قاموس شامل للألفاظ العامية الطبية
        self.slang_normalization = {
            # ألفاظ الألم العامة
            'يعورني': 'ألم',
            'يوجعني': 'ألم', 
            'تعورني': 'ألم',
            'توجعني': 'ألم',
            'يألمني': 'ألم',
            'مؤلم': 'ألم',
            
            # الحلق والتنفس
            'حلقي يلعب': 'التهاب حلق',
            'حلقي يحرق': 'التهاب حلق',
            'حنجرتي تعورني': 'التهاب حلق',
            'صدري يسكر': 'ضيق تنفس',
            'صدري ضيق': 'ضيق تنفس',
            'ما أقدر أتنفس': 'ضيق تنفس',
            'نفسي قاطع': 'ضيق تنفس',
            
            # البطن والمعدة
            'بطني يلوي': 'مغص',
            'بطني يعورني': 'ألم معدة',
            'معدتي تعورني': 'ألم معدة',
            'بطني ملوي': 'مغص',
            'أحس بلويان': 'مغص',
            'كرشي يعورني': 'ألم معدة',
            
            # الرأس والعيون
            'راس ثقيل': 'صداع',
            'راسي ثقيل': 'صداع',
            'رأسي ثقيل': 'صداع',
            'راسي يعورني': 'صداع',
            'رأسي يعورني': 'صداع',
            'عيوني تحرق': 'حساسية عيون',
            'عيني تدمع': 'حساسية عيون',
            'عيوني حمراء': 'التهاب عيون',
            
            # السعال والزكام
            'كحه': 'سعال',
            'كحة': 'سعال',
            'يكح': 'سعال',
            'اسعل': 'سعال',
            'اسعال': 'سعال',
            'أكح': 'سعال',
            'انفي مسدود': 'احتقان',
            'انفي سايل': 'رشح',
            'مزكوم': 'زكام',
            
            # الحرارة والحمى
            'عندي سخونة': 'حمى',
            'حار': 'حمى',
            'محموم': 'حمى',
            'سخن': 'حمى',
            
            # أعراض أخرى
            'يلوع': 'غثيان',
            'أبي أتقيأ': 'غثيان',
            'دايخ': 'دوخة',
            'دوخان': 'دوخة',
            'تعبان': 'تعب عام',
            'مكسر': 'تعب عام',
            'مرهق': 'تعب عام'
        }
        
        # نظام Triage واضح ومحدد
        self.emergency_symptoms = {
            'ar': [
                # صعوبات التنفس - أولوية قصوى
                'ضيق نفس', 'ضيقة نفس', 'صعوبة تنفس', 'صعوبة في التنفس', 'اختناق',
                'صدري يسكر', 'ما أقدر أتنفس', 'نفسي قاطع', 'أختنق',
                
                # ألم الصدر والقلب
                'ألم صدر شديد', 'ألم في القلب', 'خفقان شديد', 'صدري يعورني قوي',
                'أحس بضغط في صدري', 'ألم يمتد للذراع',
                
                # فقدان الوعي والتشنجات
                'إغماء', 'فقدان وعي', 'تشنجات', 'تشنج', 'نوبة', 'رجفة شديدة',
                'سقطت مغشي عليه', 'أغمي عليه', 'نوبة صرع',
                
                # الحساسية الشديدة
                'تورم الوجه', 'تورم الشفاه', 'تورم في الوجه', 'انتفاخ الوجه',
                'طفح جلدي شديد', 'طفح شديد', 'حساسية شديدة', 'حكة شديدة في كل الجسم',
                'طفح أحمر منتشر', 'جلدي كله أحمر',
                
                # أعراض عصبية
                'فقدان القدرة على الكلام', 'لا أستطيع الكلام', 'صعوبة في الكلام',
                'فقدان القدرة على الحركة', 'شلل', 'لا أستطيع الحركة',
                'خدر في نصف الجسم', 'وجهي منحرف',
                
                # نزيف وقيء شديد
                'قيء شديد مستمر', 'استفراغ مستمر', 'تقيؤ لا يتوقف',
                'نزيف شديد', 'دم كثير', 'نزف', 'قيء دم', 'براز أسود'
            ],
            'en': [
                'shortness of breath', 'difficulty breathing', 'cant breathe', 'choking',
                'chest pain', 'heart pain', 'severe palpitations',
                'fainting', 'unconscious', 'seizures', 'convulsions', 'fits',
                'facial swelling', 'lip swelling', 'face swollen',
                'severe rash', 'severe allergy', 'severe itching all over',
                'cannot speak', 'difficulty speaking', 'speech problems',
                'cannot move', 'paralysis', 'weakness', 'numbness',
                'severe vomiting', 'continuous vomiting', 'wont stop vomiting',
                'severe bleeding', 'heavy bleeding', 'vomiting blood'
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

        # قاموس شامل لأسماء الأدوية التجارية
        self.drug_synonyms = {
            # باراسيتامول
            'فيفادول': 'paracetamol',
            'بندول': 'paracetamol',
            'بنادول': 'paracetamol',
            'أدول': 'paracetamol',
            'تايلينول': 'paracetamol',
            'سيتال': 'paracetamol',
            'سيتامول': 'paracetamol',
            'نوفالدول': 'paracetamol',
            'أكامول': 'paracetamol',
            'ريفانين': 'paracetamol',
            'panadol': 'paracetamol',
            'fevadol': 'paracetamol',
            'adol': 'paracetamol',
            'tylenol': 'paracetamol',
            'novaldol': 'paracetamol',
            'acamol': 'paracetamol',
            
            # إيبوبروفين
            'بروفين': 'ibuprofen',
            'أدفيل': 'ibuprofen',
            'نوروفين': 'ibuprofen',
            'بلفين': 'ibuprofen',
            'موترين': 'ibuprofen',
            'إيبوفين': 'ibuprofen',
            'فلدين': 'ibuprofen',
            'profin': 'ibuprofen',
            'advil': 'ibuprofen',
            'nurofen': 'ibuprofen',
            'motrin': 'ibuprofen',
            'brufen': 'ibuprofen',
            
            # مضادات الحساسية
            'كلاريتين': 'loratadine',
            'تيلفاست': 'fexofenadine',
            'زيرتك': 'cetirizine',
            'أليرجيل': 'cetirizine',
            'هيستوب': 'cetirizine',
            'claritine': 'loratadine',
            'telfast': 'fexofenadine',
            'zyrtec': 'cetirizine',
            'allergyl': 'cetirizine',
            
            # أدوية البرد والسعال
            'ديكول': 'dextromethorphan',
            'فلوتاب': 'paracetamol',  # مركب
            'كومتريكس': 'paracetamol',  # مركب
            'نايت كولد': 'paracetamol',  # مركب
            'ديكونجستيل': 'dextromethorphan',
            'decol': 'dextromethorphan',
            'fluotab': 'paracetamol',
            'comtrex': 'paracetamol',
            'night_cold': 'paracetamol',
            
            # أخرى
            'أسبرين': 'aspirin',
            'اسبرين': 'aspirin',
            'aspirin': 'aspirin',
            'اسبوسيد': 'aspirin',
            'جوسبرين': 'aspirin',
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
        """نظام Intent Detection شامل وواضح"""
        query_lower = query.lower()
        normalized_text = self.case_classifier.symptom_parser.normalize_text(query)
        
        # 1. فحص الطوارئ أولاً
        emergency_words = self.case_classifier.symptom_parser.emergency_symptoms.get(language, [])
        if any(word in normalized_text for word in emergency_words):
            return 'emergency'
        
        # 2. فحص استفسارات الأطفال
        child_indicators = {
            'ar': ['ولدي', 'بنتي', 'طفلي', 'رضيعي', 'عمره', 'عمرها', 'شهر', 'سنة', 'طفل'],
            'en': ['my child', 'my baby', 'my son', 'my daughter', 'months old', 'years old', 'child', 'baby']
        }
        if any(word in query_lower for word in child_indicators.get(language, [])):
            return 'child_inquiry'
        
        # 3. فحص وجود أسماء أدوية محددة
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(query)
        has_drug_names = len(detected_drugs) > 0
        
        # 4. فحص استفسارات البدائل
        alternative_questions = {
            'ar': ['بديل', 'بدل', 'ما لقيت', 'غير متوفر', 'نفس المفعول', 'مثل'],
            'en': ['alternative', 'substitute', 'instead of', 'replacement', 'same effect', 'similar to']
        }
        if any(phrase in query_lower for phrase in alternative_questions.get(language, [])):
            return 'alternative_request'
        
        # 5. فحص استفسارات التداخل
        interaction_questions = {
            'ar': ['مع بعض', 'تداخل', 'آمن مع', 'يتعارض', 'أخذ مع', 'جمع بين'],
            'en': ['together', 'with', 'interaction', 'safe with', 'combine', 'take with']
        }
        if any(phrase in query_lower for phrase in interaction_questions.get(language, [])):
            return 'interaction_check'
        
        # 6. فحص استفسارات الجرعة
        dose_questions = {
            'ar': ['كم الجرعة', 'كيف آخذ', 'كم مرة', 'جرعة', 'كم حبة', 'كمية'],
            'en': ['how much', 'dosage', 'how many', 'dose', 'how often', 'quantity']
        }
        if any(phrase in query_lower for phrase in dose_questions.get(language, [])):
            return 'dose_inquiry'
        
        # 7. فحص وجود أعراض
        all_symptoms = (self.case_classifier.symptom_parser.normal_symptoms.get(language, []) +
                       self.case_classifier.symptom_parser.needs_info_symptoms.get(language, []))
        has_symptoms = any(word in normalized_text for word in all_symptoms)
        
        # 8. فحص الأسئلة العامة عن الأدوية
        drug_general_questions = {
            'ar': ['دواء للصداع', 'دواء للحمى', 'دواء للسعال', 'علاج للزكام', 'وش فايدة', 'معلومات عن'],
            'en': ['medicine for headache', 'medicine for fever', 'drug for', 'what is', 'information about']
        }
        has_drug_question = any(phrase in query_lower for phrase in drug_general_questions.get(language, []))
        
        # المنطق النهائي
        if has_symptoms and not has_drug_names:
            return 'symptom_only'
        elif has_drug_names and not has_symptoms:
            return 'drug_info'
        elif has_symptoms and has_drug_names:
            return 'mixed_symptom_drug'
        elif has_drug_question:
            return 'drug_general_question'
        else:
            return 'unclear'

    def process_query(self, user_input: str, language: str) -> str:
        """معالجة الاستفسار مع نظام Intent Detection المحسن"""
        user_data = st.session_state.get('user_data', {})
        
        # تحديد نوع الاستفسار
        intent = self.detect_user_intent(user_input, language)
        
        # التعامل مع كل Intent بشكل مخصص
        if intent == 'emergency':
            return self.handle_emergency(user_input, language)
        elif intent == 'child_inquiry':
            return self.handle_child_inquiry(user_input, user_data, language)
        elif intent == 'symptom_only':
            return self.handle_symptom_inquiry(user_input, user_data, language)
        elif intent == 'drug_info':
            return self.handle_drug_inquiry(user_input, language)
        elif intent == 'alternative_request':
            return self.handle_alternative_request(user_input, language)
        elif intent == 'interaction_check':
            return self.handle_interaction_check(user_input, user_data, language)
        elif intent == 'dose_inquiry':
            return self.handle_dose_inquiry(user_input, user_data, language)
        elif intent == 'mixed_symptom_drug':
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

    def handle_emergency(self, user_input: str, language: str) -> str:
        """معالجة الحالات الطارئة بتحويل فوري"""
        if language == 'ar':
            return """🚨 **حالة طارئة - تحويل فوري**

توجه للمستشفى فوراً أو اتصل بالطوارئ:
📞 **الطوارئ: 997**
📞 **الإسعاف: 997**

⚠️ لا تنتظر - الوقت مهم جداً في حالتك"""
        else:
            return """🚨 **Emergency - Immediate Referral**

Go to hospital immediately or call emergency:
📞 **Emergency: 997**
📞 **Ambulance: 997**

⚠️ Don't wait - time is critical in your case"""

    def handle_child_inquiry(self, user_input: str, user_data: Dict, language: str) -> str:
        """معالجة استفسارات الأطفال مع قواعد صارمة"""
        # فحص العمر والوزن
        age = user_data.get('age', '')
        weight = user_data.get('weight', '')
        
        eligibility = self.check_pediatric_eligibility(age, weight, language)
        
        if not eligibility['eligible']:
            if eligibility.get('action') == 'refer_to_pharmacist':
                return self.create_pharmacist_referral({
                    'classification': 'pediatric_referral',
                    'message_ar': eligibility['reason_ar'],
                    'message_en': eligibility['reason_en']
                }, user_input, user_data, language)
            
            if language == 'ar':
                return f"❌ **مطلوب معلومات إضافية:**\n\n{eligibility['reason_ar']}\n\nأرجو تحديد العمر والوزن بدقة قبل المتابعة."
            else:
                return f"❌ **Additional information required:**\n\n{eligibility['reason_en']}\n\nPlease specify age and weight accurately before proceeding."
        
        # إذا كان مؤهلاً، تابع مع معالجة الأعراض
        return self.handle_symptom_inquiry(user_input, user_data, language)

    def handle_alternative_request(self, user_input: str, language: str) -> str:
        """معالجة طلبات البدائل"""
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(user_input)
        
        if not detected_drugs:
            if language == 'ar':
                return "أحتاج اسم الدواء الأصلي لأقترح لك بديل. مثال: بديل للبندول؟"
            else:
                return "I need the original drug name to suggest alternatives. Example: alternative to Panadol?"
        
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)
        
        if not drug_info:
            if language == 'ar':
                return f"معذرة، لا تتوفر معلومات عن بدائل {drug_name} في قاعدة البيانات."
            else:
                return f"Sorry, no alternative information available for {drug_name} in database."
        
        if language == 'ar':
            response = f"🔄 **بدائل {drug_info['name_ar']}:**\n\n"
            if drug_info['alternatives_ar']:
                response += "البدائل المتاحة:\n"
                for alt in drug_info['alternatives_ar'][:3]:
                    response += f"• {alt}\n"
                response += f"\n**نفس التأثير:** {drug_info['general_use_ar']}"
            else:
                response += "لا توجد بدائل مدرجة لهذا الدواء حالياً."
        else:
            response = f"🔄 **Alternatives to {drug_info['name_en']}:**\n\n"
            if drug_info['alternatives_en']:
                response += "Available alternatives:\n"
                for alt in drug_info['alternatives_en'][:3]:
                    response += f"• {alt}\n"
                response += f"\n**Same effect:** {drug_info['general_use_en']}"
            else:
                response += "No alternatives listed for this medication currently."
        
        return response

    def handle_interaction_check(self, user_input: str, user_data: Dict, language: str) -> str:
        """فحص التداخلات الدوائية"""
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(user_input)
        current_meds = user_data.get('current_medications', '')
        
        if len(detected_drugs) < 2 and not current_meds:
            if language == 'ar':
                return "لفحص التداخل، أحتاج اسماء دوائين على الأقل. مثال: هل آمن أخذ بندول مع بروفين؟"
            else:
                return "To check interactions, I need at least two drug names. Example: Is it safe to take Panadol with Profin?"
        
        if language == 'ar':
            return """⚠️ **فحص التداخلات الدوائية**

هذه خدمة متخصصة تحتاج مراجعة صيدلي مختص.

**نصائح عامة:**
• لا تأخذ دوائين مسكنين معاً
• اقرأ النشرة الداخلية دائماً
• استشر الصيدلي قبل الجمع بين أدوية

تحويل لصيدلي مختص..."""
        else:
            return """⚠️ **Drug Interaction Check**

This is a specialized service requiring expert pharmacist review.

**General tips:**
• Don't take two pain relievers together
• Always read medication leaflets
• Consult pharmacist before combining drugs

Referring to specialist pharmacist..."""

    def handle_dose_inquiry(self, user_input: str, user_data: Dict, language: str) -> str:
        """معالجة استفسارات الجرعة مع قيود صارمة"""
        detected_drugs = self.case_classifier.symptom_parser.extract_drug_names(user_input)
        
        if not detected_drugs:
            if language == 'ar':
                return "أحتاج اسم الدواء لأحدد الجرعة. مثال: كم جرعة البندول؟"
            else:
                return "I need the drug name to determine dosage. Example: What's the dose of Panadol?"
        
        # فحص إذا كان سؤال عن طفل
        age = user_data.get('age', '')
        if age:
            eligibility = self.check_pediatric_eligibility(age, user_data.get('weight', ''), language)
            if not eligibility['eligible']:
                return self.handle_child_inquiry(user_input, user_data, language)
        
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)
        
        if not drug_info:
            if language == 'ar':
                return f"معذرة، لا تتوفر معلومات جرعة {drug_name} في قاعدة البيانات."
            else:
                return f"Sorry, dosage information for {drug_name} not available in database."
        
        if language == 'ar':
            response = f"💊 **جرعة {drug_info['name_ar']}:**\n\n"
            response += f"التراكيز المتوفرة: {', '.join(drug_info['concentrations'])}\n\n"
            response += "**للبالغين:** استشر الصيدلي للجرعة المحددة\n"
            response += "**للأطفال:** يتطلب عمر ووزن دقيق\n\n"
            response += "⚠️ **مهم:** الجرعة تختلف حسب العمر والوزن والحالة"
        else:
            response = f"💊 **{drug_info['name_en']} Dosage:**\n\n"
            response += f"Available strengths: {', '.join(drug_info['concentrations'])}\n\n"
            response += "**Adults:** Consult pharmacist for specific dose\n"
            response += "**Children:** Requires precise age and weight\n\n"
            response += "⚠️ **Important:** Dosage varies by age, weight, and condition"
        
        return response

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

    def check_pediatric_eligibility(self, age_str: str, weight_str: str, language: str) -> Dict:
        """فحص إمكانية إعطاء جرعات الأطفال مع شروط صارمة"""
        # الشرط الأول: العمر يجب أن يكون واضحاً
        if not age_str or not age_str.strip():
            return {
                'eligible': False,
                'reason_ar': 'العمر غير واضح - مطلوب تحديد عمر الطفل بدقة',
                'reason_en': 'Age not clear - precise age required for child'
            }
        
        # الشرط الثاني: الوزن يجب أن يكون واضحاً
        if not weight_str or not weight_str.strip():
            return {
                'eligible': False,
                'reason_ar': 'الوزن غير واضح - مطلوب تحديد وزن الطفل',
                'reason_en': 'Weight not clear - child weight required'
            }
        
        try:
            # استخراج العمر
            age_match = re.findall(r'(\d+)', age_str)
            if not age_match:
                return {
                    'eligible': False,
                    'reason_ar': 'صيغة العمر غير واضحة',
                    'reason_en': 'Age format unclear'
                }
            
            age_num = int(age_match[0])
            
            # تحديد العمر بالشهور
            if 'شهر' in age_str or 'month' in age_str.lower():
                age_months = age_num
            else:  # سنوات
                age_months = age_num * 12
            
            # القاعدة الصارمة: أقل من سنتين = تحويل
            if age_months < 24:
                return {
                    'eligible': False,
                    'reason_ar': 'الأطفال أقل من سنتين يحتاجون استشارة صيدلي مختص',
                    'reason_en': 'Children under 2 years need specialist pharmacist consultation',
                    'action': 'refer_to_pharmacist'
                }
            
            # استخراج الوزن
            weight_match = re.findall(r'(\d+\.?\d*)', weight_str)
            if not weight_match:
                return {
                    'eligible': False,
                    'reason_ar': 'صيغة الوزن غير واضحة',
                    'reason_en': 'Weight format unclear'
                }
            
            return {
                'eligible': True,
                'age_months': age_months,
                'weight': float(weight_match[0]),
                'age_str': age_str,
                'weight_str': weight_str
            }
            
        except Exception:
            return {
                'eligible': False,
                'reason_ar': 'خطأ في معالجة البيانات',
                'reason_en': 'Data processing error'
            }

    def handle_unclear_query(self, user_input: str, language: str) -> str:
        """سؤال واحد واضح بدلاً من "لم أفهم" """
        query_lower = user_input.lower()
        normalized_text = self.case_classifier.symptom_parser.normalize_text(user_input)
        
        # تخمين ذكي وسؤال محدد
        if language == 'ar':
            # التحيات
            if any(word in query_lower for word in ['مرحبا', 'هلا', 'السلام']):
                return "أهلاً! هل تسأل عن دواء معين أو عندك عرض معين؟"
            
            # الشكر
            elif any(word in query_lower for word in ['شكراً', 'شكرا', 'يعطيك العافية']):
                return "العفو! عندك سؤال ثاني؟"
            
            # تخمين من كلمات مفتاحية
            elif 'ألم' in normalized_text or 'يعور' in normalized_text:
                return "هل تقصد ألم في مكان معين؟ مثلاً: راسي يعورني؟"
            
            elif 'حرارة' in normalized_text or 'سخونة' in normalized_text:
                return "هل تقصد ارتفاع في الحرارة؟ وكم العمر؟"
            
            elif any(word in query_lower for word in ['دواء', 'علاج', 'حبوب']):
                return "هل تسأل عن دواء معين؟ أو تبغى دواء لعرض معين؟"
            
            elif any(word in query_lower for word in ['طفل', 'ولد', 'بنت']):
                return "هل تسأل عن طفل؟ كم العمر والوزن؟"
            
            else:
                return "هل تسأل عن دواء معين أو عندك عرض تبغى له علاج؟"
        
        else:  # English
            if any(word in query_lower for word in ['hello', 'hi', 'hey']):
                return "Hello! Are you asking about a specific medicine or do you have symptoms?"
            
            elif any(word in query_lower for word in ['thank', 'thanks']):
                return "You're welcome! Any other questions?"
            
            elif 'pain' in query_lower:
                return "Do you mean pain in a specific area? Like: I have a headache?"
            
            elif 'fever' in query_lower:
                return "Do you mean high temperature? What's the age?"
            
            elif any(word in query_lower for word in ['medicine', 'drug', 'medication']):
                return "Are you asking about a specific medicine or need medicine for symptoms?"
            
            elif any(word in query_lower for word in ['child', 'baby', 'kid']):
                return "Are you asking about a child? What's the age and weight?"
            
            else:
                return "Are you asking about a specific medicine or do you have symptoms that need treatment?"

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
        # Check if there's a selected example to populate the text area
        example_value = st.session_state.get('selected_example', '')
        if example_value:
            # Clear the selected example after using it
            st.session_state.selected_example = ''
        
        user_input = st.text_area("اكتب رسالتك (عربي/إنجليزي):", 
                                 value=example_value,
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
                # Store the example in session state for the next render
                st.session_state.selected_example = example
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

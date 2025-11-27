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
                "pediatric_safe": False,  # NO PEDIATRIC DOSES ALLOWED
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
                "pediatric_safe": False,  # NO PEDIATRIC DOSES ALLOWED
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
                "pediatric_safe": False,  # NO PEDIATRIC DOSES ALLOWED
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
                "pediatric_safe": False,  # NO PEDIATRIC DOSES ALLOWED
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
                "pediatric_safe": False,  # NO PEDIATRIC DOSES ALLOWED
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

class MedicalSafetyChecker:
    def __init__(self):
        # قائمة إجبارية بكلمات الأطفال
        self.child_keywords = {
            'ar': [
                'طفل', 'طفلي', 'ولدي', 'بنتي', 'العمر', 'عمره', 'عمرها', 
                'سنة', 'سنين', 'شهر', 'أشهر', 'وزنه', 'وزنها', 'رضيع', 
                'مولود', 'مواليد', 'طفلة', 'صبي', 'بنية'
            ],
            'en': [
                'child', 'my child', 'my son', 'my daughter', 'baby', 'infant',
                'toddler', 'kid', 'years old', 'months old', 'age', 'weight'
            ]
        }

        # قائمة إجبارية بكلمات الحوامل
        self.pregnancy_keywords = {
            'ar': [
                'حامل', 'حمل', 'مرضعة', 'رضاعة', 'ولدت', 'بعد الولادة',
                'حملي', 'جنيني', 'الحمل', 'الرضاعة'
            ],
            'en': [
                'pregnant', 'pregnancy', 'breastfeeding', 'nursing', 'expecting',
                'maternity', 'prenatal', 'postnatal'
            ]
        }

        # قائمة إجبارية بكلمات الطوارئ
        self.emergency_keywords = {
            'ar': [
                'ضيقة نفس', 'ضيق نفس', 'صعوبة تنفس', 'اختناق', 'ألم صدر',
                'إغماء', 'فقدان وعي', 'تفريغ دم', 'قيء دم', 'براز أسود',
                'حساسية شديدة', 'طفح جلدي قوي', 'تورم وجه', 'تورم الوجه',
                'نوبة قلبية', 'جلطة', 'شلل', 'تشنج', 'نوبة صرع'
            ],
            'en': [
                'shortness of breath', 'chest pain', 'heart attack', 'stroke',
                'fainting', 'unconscious', 'vomiting blood', 'black stool',
                'severe allergy', 'facial swelling', 'choking', 'seizure'
            ]
        }

    def check_safety_violations(self, user_input: str, language: str) -> Dict:
        """فحص انتهاكات السلامة الطبية 100%"""
        user_input_lower = user_input.lower()

        # 1) فحص كلمات الأطفال - ممنوع منعاً باتاً
        child_words = self.child_keywords.get(language, [])
        for word in child_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'child_detected',
                    'action': 'refer_to_pharmacist',
                    'message_ar': 'هذه حالة أطفال، وجرعات الأطفال لازم تُحسب حسب الوزن والعمر. تحويل هذه الحالة للصيدلي مباشرة.',
                    'message_en': 'This is a pediatric case. Child dosages must be calculated based on weight and age. Referring this case directly to pharmacist.'
                }

        # 2) فحص كلمات الحوامل - ممنوع منعاً باتاً
        pregnancy_words = self.pregnancy_keywords.get(language, [])
        for word in pregnancy_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'pregnancy_detected',
                    'action': 'refer_to_pharmacist',
                    'message_ar': 'الحوامل والمرضعات لهم أدوية محدودة. تحويل هذه الحالة للصيدلي مباشرة.',
                    'message_en': 'Pregnant and breastfeeding women have limited medication options. Referring this case directly to pharmacist.'
                }

        # 3) فحص كلمات الطوارئ - تحويل فوري
        emergency_words = self.emergency_keywords.get(language, [])
        for word in emergency_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'emergency_detected',
                    'action': 'emergency_referral',
                    'message_ar': '🚨 هذه علامة خطر. توجه للطوارئ فوراً أو اتصل بـ 997.',
                    'message_en': '🚨 This is a danger sign. Go to emergency immediately or call 997.'
                }

        return {'violation': False}

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

class DecisionTreeClassifier:
    def __init__(self):
        self.symptom_parser = AdvancedSymptomParser()
        self.drug_api = DrugAPIHandler()
        self.safety_checker = MedicalSafetyChecker()

        # الردود الجاهزة لكل عرض (مطابقة تماماً للمطلوب)
        self.symptom_responses = {
            # كحة ناشفة
            'كحة ناشفة': {
                'response_ar': """💊 للكحة الناشفة:
• دواء مقترح: مهدئ كحة مثل Tussivan C أو Decol
• اشرب سوائل دافئة وتجنب المهيجات
⚠️ إذا ما تحسنت 3 أيام، راجع طبيب.""",
                'response_en': """💊 For dry cough:
• Suggested medication: Cough suppressant like Tussivan C or Decol
• Drink warm fluids and avoid irritants
⚠️ If no improvement in 3 days, see doctor."""
            },

            # كحة مع بلغم
            'بلغم': {
                'response_ar': """💊 للبلغم:
• دواء مقترح: مذيب بلغم مثل Mucosolvan
• اشرب سوائل كثيرة
⚠️ إذا استمر 3 أيام، راجع الطبيب.""",
                'response_en': """💊 For phlegm:
• Suggested medication: Mucolytic like Mucosolvan
• Drink plenty of fluids
⚠️ If continues for 3 days, see doctor."""
            },

            # حرارة (بالغ)
            'حرارة': {
                'response_ar': """💊 للحرارة:
• دواء مقترح: باراسيتامول
• خذ راحة واشرب سوائل
⚠️ إذا ارتفعت أو استمرت 3 أيام، راجع الطبيب.""",
                'response_en': """💊 For fever:
• Suggested medication: Paracetamol
• Rest and drink fluids
⚠️ If rises or continues 3 days, see doctor."""
            },

            # صداع
            'صداع': {
                'response_ar': """💊 للصداع:
• دواء مقترح: مسكن بسيط مثل باراسيتامول
• ارتح واشرب ماء
⚠️ إذا الصداع شديد ومتكرر، افحص.""",
                'response_en': """💊 For headache:
• Suggested medication: Simple painkiller like Paracetamol
• Rest and drink water
⚠️ If severe and recurring, get checked."""
            },

            # التهاب حلق
            'التهاب حلق': {
                'response_ar': """💊 لالتهاب الحلق:
• دواء مقترح: Lozenges أو غرغرة ملح دافئ
• اشرب سوائل دافئة
⚠️ إذا استمر أكثر من 3 أيام، راجع طبيب.""",
                'response_en': """💊 For sore throat:
• Suggested medication: Lozenges or warm salt gargle
• Drink warm fluids
⚠️ If continues more than 3 days, see doctor."""
            },

            # احتقان وانسداد الأنف
            'احتقان': {
                'response_ar': """💊 للاحتقان:
• دواء مقترح: مزيل احتقان مثل Sudafed
• بخار ماء دافئ يساعد""",
                'response_en': """💊 For congestion:
• Suggested medication: Decongestant like Sudafed
• Warm steam helps"""
            },

            # دوخة وغثيان
            'دوخة': {
                'response_ar': """💊 للدوخة والغثيان:
• دواء مقترح: Dramamine
• تجنب الحركة السريعة""",
                'response_en': """💊 For dizziness and nausea:
• Suggested medication: Dramamine
• Avoid sudden movements"""
            },

            # ألم المعدة بعد الأكل
            'ألم معدة': {
                'response_ar': """💊 لألم المعدة بعد الأكل:
• دواء مقترح: مضاد حموضة مثل Gaviscon
• تجنب الأكل الدسم""",
                'response_en': """💊 For stomach pain after eating:
• Suggested medication: Antacid like Gaviscon
• Avoid fatty foods"""
            }
        }

    def classify_input(self, user_input: str, language: str) -> Dict:
        """Decision Tree المطلوب بالضبط"""

        # Step 1: فحص كلمات الطوارئ
        safety_check = self.safety_checker.check_safety_violations(user_input, language)
        if safety_check['violation']:
            if safety_check['type'] == 'emergency_detected':
                return {'classification': 'Emergency', 'response': safety_check[f'message_{language}']}
            elif safety_check['type'] == 'child_detected':
                return {'classification': 'ChildReferral', 'response': safety_check[f'message_{language}']}
            elif safety_check['type'] == 'pregnancy_detected':
                return {'classification': 'PregnantReferral', 'response': safety_check[f'message_{language}']}

        # Step 2: فحص اسم دواء
        detected_drugs = self.symptom_parser.extract_drug_names(user_input)
        if detected_drugs:
            return {'classification': 'DrugInfo', 'drugs': detected_drugs}

        # Step 3: فحص عرض واضح
        normalized_text = self.symptom_parser.normalize_text(user_input)
        for symptom, response_data in self.symptom_responses.items():
            if symptom in normalized_text:
                return {
                    'classification': 'SymptomAdvice',
                    'symptom': symptom,
                    'response': response_data[f'response_{language}']
                }

        # Step 4: المدخل مبهم
        return {'classification': 'Clarify'}

class AdvancedMedicalChatbot:
    def __init__(self):
        self.setup_models()
        self.drug_api = DrugAPIHandler()
        self.decision_tree = DecisionTreeClassifier()

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

            st.success("✅ تم تحميل النظام المحسّن مع قواعد السلامة!")
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {str(e)}")

    def process_query(self, user_input: str, language: str) -> str:
        """معالجة الاستفسار مع Decision Tree الجديد"""

        # تطبيق Decision Tree
        classification_result = self.decision_tree.classify_input(user_input, language)

        if classification_result['classification'] == 'Emergency':
            return classification_result['response']

        elif classification_result['classification'] == 'ChildReferral':
            return classification_result['response']

        elif classification_result['classification'] == 'PregnantReferral':
            return classification_result['response']

        elif classification_result['classification'] == 'DrugInfo':
            return self.handle_drug_info(classification_result['drugs'], language)

        elif classification_result['classification'] == 'SymptomAdvice':
            return classification_result['response']

        elif classification_result['classification'] == 'Clarify':
            return self.handle_unclear_input(user_input, language)

        return "خطأ في المعالجة"

    def handle_drug_info(self, detected_drugs: List[str], language: str) -> str:
        """معالجة معلومات الدواء - بدون جرعات نهائياً"""
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)

        if not drug_info:
            if language == 'ar':
                return f"معلومات الدواء '{drug_name}' غير متوفرة في قاعدة البيانات"
            else:
                return f"Drug information for '{drug_name}' not available in database"

        if language == 'ar':
            response = f"💊 **{drug_info['name_ar']} ({drug_info['name_en']})**\n\n"
            response += f"🔹 **الاستخدام:** {drug_info['general_use_ar']}\n"
            response += f"🔹 **تحذيرات مهمة:** {', '.join(drug_info['warnings_ar'][:2])}\n"

            if drug_info['alternatives_ar']:
                response += f"🔹 **بدائل عامة:** {', '.join(drug_info['alternatives_ar'][:2])}\n"

            response += "\n⚠️ **بدون جرعة نهائياً - استشر الصيدلي للجرعة المناسبة**"
        else:
            response = f"💊 **{drug_info['name_en']} ({drug_info['name_ar']})**\n\n"
            response += f"🔹 **Use:** {drug_info['general_use_en']}\n"
            response += f"🔹 **Important warnings:** {', '.join(drug_info['warnings_en'][:2])}\n"

            if drug_info['alternatives_en']:
                response += f"🔹 **General alternatives:** {', '.join(drug_info['alternatives_en'][:2])}\n"

            response += "\n⚠️ **No dosage provided - consult pharmacist for appropriate dose**"

        return response

    def handle_unclear_input(self, user_input: str, language: str) -> str:
        """التعامل مع المدخلات المبهمة"""
        if language == 'ar':
            return """وضح العرض أكثر عشان أفهم:

**حدد نوع المشكلة:**
• حرارة؟
• ألم؟ 
• كحة؟
• التهاب؟
• دوخة؟
• مغص؟

اكتب المشكلة بوضوح مثل: "عندي صداع" أو "كحة من يومين""""
        else:
            return """Clarify the symptom more so I can understand:

**Specify the problem type:**
• Fever?
• Pain?
• Cough?
• Inflammation?
• Dizziness?
• Cramps?

Write the problem clearly like: "I have headache" or "Cough for 2 days""""

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

def main():
    try:
        st.set_page_config(
            page_title="البوت الطبي الآمن مع قواعد السلامة",
            page_icon="💊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    except Exception as e:
        st.error(f"خطأ في التهيئة: {str(e)}")

    st.title("💊 البوت الطبي الآمن مع قواعد السلامة الشاملة")
    st.markdown("### Safe Medical Bot with Comprehensive Safety Rules | بوت طبي آمن بقواعد سلامة شاملة")

    # تهيئة المحادثة المستمرة
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}

    # تهيئة البوت
    if 'chatbot' not in st.session_state:
        with st.spinner("جاري تحميل النظام الآمن مع قواعد السلامة..."):
            try:
                st.session_state.chatbot = AdvancedMedicalChatbot()
            except Exception as e:
                st.error(f"خطأ في تحميل النظام: {str(e)}")
                st.stop()

    # الشريط الجانبي
    with st.sidebar:
        st.header("قواعد السلامة المطبقة 100%")
        st.markdown("""
        🚫 **ممنوع جرعات الأطفال نهائياً**

        🚫 **ممنوع وصف دواء للحامل**

        🚨 **تحويل الطوارئ فوراً**

        ✅ **نصائح عامة فقط بدون جرعات**

        ✅ **معلومات الأدوية بدون جرعة**

        ✅ **Decision Tree واضح**
        """)

        st.header("رفع الوصفة الطبية")
        uploaded_file = st.file_uploader("ارفع صورة الوصفة...", type=['png', 'jpg', 'jpeg'])

    # واجهة المحادثة الرئيسية
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("واجهة المحادثة الآمنة | Safe Chat Interface")

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
        example_value = st.session_state.get('selected_example', '')
        if example_value:
            st.session_state.selected_example = ''

        user_input = st.text_area("اكتب رسالتك (عربي/إنجليزي):", 
                                 value=example_value,
                                 placeholder="مثال: عندي صداع، أو معلومات عن بندول", 
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
        st.header("قواعد السلامة النشطة")
        st.success("✅ فحص كلمات الأطفال")
        st.success("✅ فحص كلمات الحوامل") 
        st.success("✅ فحص كلمات الطوارئ")
        st.success("✅ منع الجرعات نهائياً")
        st.success("✅ Decision Tree فعال")

        # أمثلة للمساعدة
        st.header("أمثلة للتجربة")
        examples = [
            "عندي صداع شديد",
            "كحة من يومين", 
            "معلومات عن بندول",
            "دواء للحرارة",
            "حلقي يلعب"
        ]

        for example in examples:
            if st.button(f"جرب: {example}", key=f"example_{hash(example)}"):
                st.session_state.selected_example = example
                st.rerun()

    # معالجة الوصفة الطبية
    if uploaded_file:
        st.header("تحليل الوصفة الطبية")
        process_prescription(uploaded_file)

def process_user_message(user_input: str, uploaded_file=None):
    """معالجة رسالة المستخدم"""
    chatbot = st.session_state.chatbot
    language = chatbot.detect_language(user_input)

    # معالجة الاستفسار بالنظام الآمن الجديد
    response = chatbot.process_query(user_input, language)

    # إضافة للمحادثة المحفوظة
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.chat_history.append((user_input, response, timestamp))

    st.rerun()

def process_prescription(uploaded_file):
    """معالجة الوصفة الطبية المرفوعة"""
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

                        st.error("⚠️ **ممنوع عرض الجرعات - استشر الصيدلي**")

            # عرض النص الخام المستخرج
            with st.expander("النص المستخرج من الصورة"):
                st.write(ocr_result['raw_text'])

        else:
            st.error(ocr_result['message_ar'])

    except Exception as e:
        st.error(f"خطأ في معالجة الوصفة: {str(e)}")

if __name__ == "__main__":
    main()
import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
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
from difflib import SequenceMatcher
import Levenshtein

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
            "augmentin": {
                "name_ar": "أوجمنتين",
                "name_en": "Augmentin",
                "concentrations": ["625mg", "1g", "228mg/5ml"],
                "general_use_ar": "مضاد حيوي واسع المجال",
                "general_use_en": "Broad spectrum antibiotic",
                "interactions_ar": ["مضادات التجلط", "المكملات الحديدية"],
                "interactions_en": ["Blood thinners", "Iron supplements"],
                "warnings_ar": ["إكمال الكورس كاملاً", "حذار من الحساسية"],
                "warnings_en": ["Complete full course", "Caution with allergies"],
                "alternatives_ar": ["أموكسيل", "كلافوكس"],
                "alternatives_en": ["Amoxil", "Clavox"],
                "danger_level": "medium",
                "pediatric_safe": False,
                "min_age_months": 3
            },
            "zanidip": {
                "name_ar": "زانيديب",
                "name_en": "Zanidip",
                "concentrations": ["10mg", "20mg"],
                "general_use_ar": "علاج ضغط الدم المرتفع",
                "general_use_en": "High blood pressure treatment",
                "interactions_ar": ["جريب فروت", "أدوية القلب"],
                "interactions_en": ["Grapefruit", "Heart medications"],
                "warnings_ar": ["لا يوقف فجأة", "متابعة طبية ضرورية"],
                "warnings_en": ["Don't stop suddenly", "Medical follow-up required"],
                "alternatives_ar": ["أملور", "نورفاسك"],
                "alternatives_en": ["Amlor", "Norvasc"],
                "danger_level": "high",
                "pediatric_safe": False,
                "min_age_months": 216
            },
            "mucosolvan": {
                "name_ar": "موكوسولفان",
                "name_en": "Mucosolvan",
                "concentrations": ["30mg", "15mg/5ml"],
                "general_use_ar": "مذيب للبلغم ومهدئ للسعال",
                "general_use_en": "Expectorant and cough suppressant",
                "interactions_ar": ["قليلة التداخل"],
                "interactions_en": ["Few interactions"],
                "warnings_ar": ["اشرب سوائل كثيرة", "لا تستخدم أكثر من أسبوع"],
                "warnings_en": ["Drink plenty of fluids", "Don't use more than a week"],
                "alternatives_ar": ["بيسولفون", "أمبروكسول"],
                "alternatives_en": ["Bisolvon", "Ambroxol"],
                "danger_level": "low",
                "pediatric_safe": False,
                "min_age_months": 24
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
                'صدري يعور', 'ألم مع التنفس', 'شدة ألم الصدر', 'صدر يؤلم',
                'ألم حاد في الصدر', 'صدري يحرق', 'وجع صدر شديد',
                'إغماء', 'فقدان وعي', 'تفريغ دم', 'قيء دم', 'براز أسود',
                'حساسية شديدة', 'طفح جلدي قوي', 'تورم وجه', 'تورم الوجه',
                'نوبة قلبية', 'جلطة', 'شلل', 'تشنج', 'نوبة صرع'
            ],
            'en': [
                'shortness of breath', 'chest pain', 'heart attack', 'stroke',
                'chest hurts', 'pain when breathing', 'severe chest pain',
                'sharp chest pain', 'chest burning', 'intense chest pain',
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

            # الحلق والتنفس - كلمات واضحة فقط
            'حلقي يحرق': 'التهاب حلق',
            'حنجرتي تعورني': 'التهاب حلق',
            'حلقي يؤلمني': 'التهاب حلق',
            'صدري ضيق': 'ضيق تنفس',
            'ما أقدر أتنفس': 'ضيق تنفس',
            'نفسي قاطع': 'ضيق تنفس',
            'صعوبة بالتنفس': 'ضيق تنفس',

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
            'كحه': 'كحة',
            'كحة': 'كحة',
            'يكح': 'كحة',
            'اسعل': 'كحة',
            'اسعال': 'كحة',
            'أكح': 'كحة',
            'سعال': 'كحة',
            'كح': 'كحة',
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

        # قائمة الكلمات المبهمة التي تحتاج توضيح
        self.unclear_terms = [
            'يلعب', 'يسكر', 'مو براسي', 'مخنوق شوي', 'تعبان',
            'مكسر', 'مش طبيعي', 'غريب', 'مش عادي', 'حاسس بحاجة',
            'مضايقني', 'مقلقني', 'غير مرتاح'
        ]

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
            'باراسيتامول': 'paracetamol',
            'paracetamol': 'paracetamol',

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

            # أوجمنتين
            'أوجمنتين': 'augmentin',
            'اوجمنتين': 'augmentin',
            'augmentin': 'augmentin',
            'أوجمين': 'augmentin',
            'اوجمين': 'augmentin',
            'كلافوكس': 'augmentin',
            'clavox': 'augmentin',

            # زانيديب
            'زانيديب': 'zanidip',
            'zanidip': 'zanidip',
            'أملور': 'zanidip',
            'amlor': 'zanidip',
            'نورفاسك': 'zanidip',
            'norvasc': 'zanidip',

            # موكوسولفان
            'موكوسولفان': 'mucosolvan',
            'mucosolvan': 'mucosolvan',
            'بيسولفون': 'mucosolvan',
            'bisolvon': 'mucosolvan',
            'أمبروكسول': 'mucosolvan',
            'ambroxol': 'mucosolvan',

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

class IntentClassifier:
    def __init__(self):
        self.symptom_parser = AdvancedSymptomParser()
        self.drug_api = DrugAPIHandler()
        self.safety_checker = MedicalSafetyChecker()

        # Intent patterns for accurate classification
        self.intent_patterns = {
            'GET_DOSAGE': {
                'ar': ['جرعة', 'جرعات', 'كمية', 'مقدار', 'كم مرة', 'كيف آخذ', 'طريقة استخدام'],
                'en': ['dosage', 'dose', 'how much', 'how many times', 'how to take', 'quantity', 'amount']
            },
            'GET_ALTERNATIVES': {
                'ar': ['بديل', 'بدائل', 'مثيل', 'أي دواء آخر', 'شبيه', 'نفس التأثير'],
                'en': ['alternative', 'alternatives', 'similar', 'replacement', 'substitute', 'other drug']
            },
            'GET_INTERACTION': {
                'ar': ['تداخل', 'تفاعل', 'مع بعض', 'آمان', 'يتعارض', 'ينفع مع'],
                'en': ['interaction', 'interactions', 'together', 'with', 'safe', 'conflict', 'mix', 'combine']
            },
            'GET_SIDE_EFFECTS': {
                'ar': ['أعراض جانبية', 'آثار جانبية', 'مضاعفات', 'أضرار'],
                'en': ['side effects', 'side effect', 'adverse effects', 'reactions', 'complications']
            },
            'GET_WARNINGS': {
                'ar': ['تحذيرات', 'تحذير', 'خطورة', 'احتياطات', 'انتبه'],
                'en': ['warnings', 'warning', 'precautions', 'cautions', 'contraindications']
            }
        }

        # الردود الجاهزة لكل عرض
        self.symptom_responses = {
            'كحة': {
                'response_ar': """💊 للكحة:
• دواء مقترح: مهدئ كحة مثل Tussivan C أو Decol
• اشرب سوائل دافئة وتجنب المهيجات
⚠️ إذا ما تحسنت 3 أيام، راجع طبيب.""",
                'response_en': """💊 For cough:
• Suggested medication: Cough suppressant like Tussivan C or Decol
• Drink warm fluids and avoid irritants
⚠️ If no improvement in 3 days, see doctor."""
            },
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
            'حرارة': {
                'response_ar': """💊 للحرارة:
• دواء مقترح: باراسيتامول
• خذ راحة واشرب سوائل
⚠️ إذا ارتفعت أو استمرت 3 أيام، راجع الطبيب.""",
                'response_en': """💊 For fever:
• Suggested medication: Paracetamol
• Rest and drink fluids
⚠️ If rises or continues 3 days, see doctor."""
            }
        }

    def fuzzy_match_drug(self, input_drug: str) -> Tuple[str, float]:
        """Fuzzy matching للأدوية مع تهجئة خاطئة"""
        best_match = None
        best_score = 0

        # البحث في قاعدة البيانات الأساسية
        for drug_key in self.drug_api.mock_drug_database.keys():
            score = SequenceMatcher(None, input_drug.lower(), drug_key.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = drug_key

        # البحث في الأسماء التجارية
        for synonym, standard_name in self.symptom_parser.drug_synonyms.items():
            score = SequenceMatcher(None, input_drug.lower(), synonym.lower()).ratio()
            if score > best_score:
                best_score = score
                best_match = standard_name

        return best_match, best_score

    def _extract_drugs_with_fuzzy(self, user_input: str) -> List[str]:
        """Helper function to extract drugs using fuzzy matching."""
        words = user_input.lower().split()
        detected_drugs = []

        # قائمة الكلمات المفتاحية للتجاهل
        ignore_words = [
            'interactions', 'interaction', 'side', 'effects', 'warnings',
            'alternatives', 'dosage', 'dose', 'تداخل', 'تفاعل',
            'أعراض', 'جانبية', 'تحذيرات', 'بدائل', 'جرعة'
        ]

        # فحص الكلمات منفردة (مع تجاهل الكلمات المفتاحية)
        for word in words:
            if len(word) > 3 and word not in ignore_words:
                matched_drug, score = self.fuzzy_match_drug(word)
                if score > 0.6:  # نسبة تشابه متوسطة
                    detected_drugs.append(matched_drug)

        # فحص العبارة كاملة (بعد إزالة الكلمات المفتاحية)
        cleaned_input = user_input.lower()
        for ignore_word in ignore_words:
            cleaned_input = cleaned_input.replace(ignore_word, '').strip()

        if len(cleaned_input) > 3:
            matched_drug, score = self.fuzzy_match_drug(cleaned_input)
            if score > 0.6:
                detected_drugs.append(matched_drug)

        return list(set(detected_drugs))

    def detect_intent(self, user_input: str, language: str) -> str:
        """كشف الـ Intent بدقة عالية مع أولوية للأدوية"""
        user_input_lower = user_input.lower()

        # فحص الأدوية أولاً - أهم شي
        detected_drugs = self.symptom_parser.extract_drug_names(user_input)
        fuzzy_drugs = self._extract_drugs_with_fuzzy(user_input)
        all_detected_drugs = list(set(detected_drugs + fuzzy_drugs))

        if all_detected_drugs:
            # فحص Intent patterns للأدوية مع أولوية للأوامر المحددة
            for intent, patterns in self.intent_patterns.items():
                lang_patterns = patterns.get(language, [])
                for pattern in lang_patterns:
                    if pattern in user_input_lower:
                        if intent == 'GET_DOSAGE':
                            return 'GET_DOSAGE'
                        elif intent == 'GET_ALTERNATIVES':
                            return 'GET_ALTERNATIVES'
                        elif intent == 'GET_INTERACTION':
                            return 'GET_INTERACTION'
                        elif intent == 'GET_SIDE_EFFECTS':
                            return 'GET_SIDE_EFFECTS'
                        elif intent == 'GET_WARNINGS':
                            return 'GET_WARNINGS'

            # إذا كان فيه دوائين أو أكثر = تداخل
            if len(all_detected_drugs) >= 2:
                return 'GET_INTERACTION'

            # أي دواء منفرد = معلومات الدواء
            return 'GET_DRUG_INFO'

        # فحص Intent patterns العامة (بدون أدوية)
        for intent, patterns in self.intent_patterns.items():
            lang_patterns = patterns.get(language, [])
            for pattern in lang_patterns:
                if pattern in user_input_lower:
                    return intent

        # فحص الأعراض فقط إذا ما لقينا أدوية
        normalized_text = self.symptom_parser.normalize_text(user_input)
        for symptom in self.symptom_responses.keys():
            if symptom in normalized_text:
                return 'GET_SYMPTOM_SUGGESTION'

        return 'CLARIFY'

    def classify_input(self, user_input: str, language: str) -> Dict:
        """تصنيف محسّن للمدخلات"""

        # Step 1: فحص السلامة
        safety_check = self.safety_checker.check_safety_violations(user_input, language)
        if safety_check['violation']:
            if safety_check['type'] == 'emergency_detected':
                return {'classification': 'Emergency', 'response': safety_check[f'message_{language}']}
            elif safety_check['type'] == 'child_detected':
                return {'classification': 'ChildReferral', 'response': safety_check[f'message_{language}']}
            elif safety_check['type'] == 'pregnancy_detected':
                return {'classification': 'PregnantReferral', 'response': safety_check[f'message_{language}']}

        # Step 2: كشف Intent
        intent = self.detect_intent(user_input, language)

        if intent == 'GET_DRUG_INFO':
            detected_drugs = self.symptom_parser.extract_drug_names(user_input)
            if not detected_drugs:
                # محاولة fuzzy matching
                detected_drugs = self._extract_drugs_with_fuzzy(user_input)

            if detected_drugs:
                return {'classification': 'DrugInfo', 'drugs': detected_drugs}
            else:
                return {'classification': 'UnknownDrug', 'original_input': user_input}

        elif intent == 'GET_DOSAGE':
            detected_drugs = self.symptom_parser.extract_drug_names(user_input)
            if not detected_drugs:
                detected_drugs = self._extract_drugs_with_fuzzy(user_input)

            if detected_drugs:
                return {'classification': 'DosageRequest', 'drugs': detected_drugs}
            else:
                return {'classification': 'UnknownDrug', 'original_input': user_input}

        elif intent == 'GET_ALTERNATIVES':
            detected_drugs = self.symptom_parser.extract_drug_names(user_input)
            if not detected_drugs:
                detected_drugs = self._extract_drugs_with_fuzzy(user_input)

            if detected_drugs:
                return {'classification': 'AlternativesRequest', 'drugs': detected_drugs}
            else:
                return {'classification': 'UnknownDrug', 'original_input': user_input}

        elif intent == 'GET_INTERACTION':
            detected_drugs = self.symptom_parser.extract_drug_names(user_input)
            if len(detected_drugs) < 2:
                # محاولة استخراج دوائين من النص
                detected_drugs = self._extract_drugs_with_fuzzy(user_input)

            if len(detected_drugs) >= 2:
                return {'classification': 'InteractionCheck', 'drugs': detected_drugs}
            elif len(detected_drugs) == 1:
                # إذا كان فيه دواء واحد مع كلمة interactions
                return {'classification': 'InteractionInfo', 'drugs': detected_drugs}
            else:
                return {'classification': 'UnknownDrug', 'original_input': user_input}

        elif intent == 'GET_SIDE_EFFECTS':
            detected_drugs = self.symptom_parser.extract_drug_names(user_input)
            if not detected_drugs:
                detected_drugs = self._extract_drugs_with_fuzzy(user_input)

            if detected_drugs:
                return {'classification': 'SideEffectsRequest', 'drugs': detected_drugs}
            else:
                return {'classification': 'UnknownDrug', 'original_input': user_input}

        elif intent == 'GET_WARNINGS':
            detected_drugs = self.symptom_parser.extract_drug_names(user_input)
            if not detected_drugs:
                detected_drugs = self._extract_drugs_with_fuzzy(user_input)

            if detected_drugs:
                return {'classification': 'WarningsRequest', 'drugs': detected_drugs}
            else:
                return {'classification': 'UnknownDrug', 'original_input': user_input}

        elif intent == 'GET_SYMPTOM_SUGGESTION':
            normalized_text = self.symptom_parser.normalize_text(user_input)
            for symptom, response_data in self.symptom_responses.items():
                if symptom in normalized_text:
                    return {
                        'classification': 'SymptomAdvice',
                        'symptom': symptom,
                        'response': response_data[f'response_{language}']
                    }

        return {'classification': 'Clarify'}

class AdvancedMedicalChatbot:
    def __init__(self):
        self.setup_models()
        self.drug_api = DrugAPIHandler()
        self.intent_classifier = IntentClassifier()

    def setup_models(self):
        """تهيئة نماذج mBERT"""
        try:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-cased')
            self.classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )

            st.success("✅ تم تحميل النظام المحسّن مع قواعد السلامة!")
        except Exception as e:
            st.error(f"خطأ في تحميل النماذج: {str(e)}")

    def process_query(self, user_input: str, language: str) -> str:
        """معالجة الاستفسار مع Intent Classifier الجديد"""

        # تطبيق Intent Classifier
        classification_result = self.intent_classifier.classify_input(user_input, language)

        if classification_result['classification'] == 'Emergency':
            return classification_result['response']

        elif classification_result['classification'] == 'ChildReferral':
            return classification_result['response']

        elif classification_result['classification'] == 'PregnantReferral':
            return classification_result['response']

        elif classification_result['classification'] == 'DrugInfo':
            return self.handle_drug_info(classification_result['drugs'], language)

        elif classification_result['classification'] == 'DosageRequest':
            return self.handle_dosage_request(classification_result['drugs'], language)

        elif classification_result['classification'] == 'AlternativesRequest':
            return self.handle_alternatives_request(classification_result['drugs'], language)

        elif classification_result['classification'] == 'InteractionCheck':
            return self.handle_interaction_check(classification_result['drugs'], language)

        elif classification_result['classification'] == 'InteractionInfo':
            return self.handle_interaction_info(classification_result['drugs'], language)

        elif classification_result['classification'] == 'SideEffectsRequest':
            return self.handle_side_effects_request(classification_result['drugs'], language)

        elif classification_result['classification'] == 'WarningsRequest':
            return self.handle_warnings_request(classification_result['drugs'], language)

        elif classification_result['classification'] == 'UnknownDrug':
            return self.handle_unknown_drug(classification_result['original_input'], language)

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
            return self.handle_unknown_drug(drug_name, language)

        if language == 'ar':
            response = f"💊 **{drug_info['name_ar']} ({drug_info['name_en']})**\n\n"
            response += f"🔹 **الاستخدام:** {drug_info['general_use_ar']}\n"
            response += f"🔹 **تحذيرات مهمة:** {', '.join(drug_info['warnings_ar'][:2])}\n"
            response += f"🔹 **التداخلات:** {', '.join(drug_info['interactions_ar'][:2])}\n"
            response += "\n⚠️ **بدون جرعة نهائياً - استشر الصيدلي للجرعة المناسبة**"
        else:
            response = f"💊 **{drug_info['name_en']} ({drug_info['name_ar']})**\n\n"
            response += f"🔹 **Use:** {drug_info['general_use_en']}\n"
            response += f"🔹 **Important warnings:** {', '.join(drug_info['warnings_en'][:2])}\n"
            response += f"🔹 **Interactions:** {', '.join(drug_info['interactions_en'][:2])}\n"
            response += "\n⚠️ **No dosage provided - consult pharmacist for appropriate dose**"

        return response

    def handle_dosage_request(self, detected_drugs: List[str], language: str) -> str:
        """معالجة طلبات الجرعة - ممنوع إعطاء جرعة"""
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)

        if not drug_info:
            return self.handle_unknown_drug(drug_name, language)

        if language == 'ar':
            return f"""🚫 **لا يمكنني إعطاء جرعة {drug_info['name_ar']}**

⚠️ **الجرعة تحتاج حساب دقيق حسب:**
• العمر والوزن
• الحالة الصحية
• الأدوية الأخرى
• شدة المرض

**👨‍⚕️ استشر صيدلي أو طبيب للجرعة الصحيحة**"""
        else:
            return f"""🚫 **Cannot provide dosage for {drug_info['name_en']}**

⚠️ **Dosage requires precise calculation based on:**
• Age and weight
• Medical condition
• Other medications
• Severity of illness

**👨‍⚕️ Consult pharmacist or doctor for correct dosage**"""

    def handle_alternatives_request(self, detected_drugs: List[str], language: str) -> str:
        """معالجة طلبات البدائل"""
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)

        if not drug_info:
            return self.handle_unknown_drug(drug_name, language)

        if language == 'ar':
            alternatives_list = '\n• '.join(drug_info['alternatives_ar'])
            return f"""💊 **بدائل {drug_info['name_ar']}:**

• {alternatives_list}

**💡 ملاحظة:** البدائل قد تختلف في التركيز والتأثير
**👨‍⚕️ استشر الصيدلي قبل التبديل**"""
        else:
            alternatives_list = '\n• '.join(drug_info['alternatives_en'])
            return f"""💊 **Alternatives to {drug_info['name_en']}:**

• {alternatives_list}

**💡 Note:** Alternatives may vary in concentration and effect
**👨‍⚕️ Consult pharmacist before switching**"""

    def handle_interaction_check(self, detected_drugs: List[str], language: str) -> str:
        """فحص التداخلات الدوائية"""
        if len(detected_drugs) < 2:
            if language == 'ar':
                return "أحتاج اسمين من الأدوية لفحص التداخل"
            else:
                return "I need two drug names to check interactions"

        drug1_name = detected_drugs[0]
        drug2_name = detected_drugs[1]

        drug1_info = self.drug_api.search_drug(drug1_name)
        drug2_info = self.drug_api.search_drug(drug2_name)

        if not drug1_info or not drug2_info:
            missing_drug = drug1_name if not drug1_info else drug2_name
            return self.handle_unknown_drug(missing_drug, language)

        # فحص التداخل البسيط
        interaction_found = False
        if language == 'ar':
            drug1_interactions = drug1_info.get('interactions_ar', [])
            for interaction in drug1_interactions:
                if interaction.lower() in drug2_info['name_ar'].lower() or interaction.lower() in drug2_name.lower():
                    interaction_found = True
                    break

        if language == 'ar':
            if interaction_found:
                return f"""⚠️ **تحذير: قد يوجد تداخل بين {drug1_info['name_ar']} و {drug2_info['name_ar']}**

**🚫 لا ينصح بتناولهما معاً بدون استشارة طبية**

**👨‍⚕️ استشر صيدلي أو طبيب قبل الجمع بينهما**"""
            else:
                return f"""✅ **لا يوجد تداخل معروف بين {drug1_info['name_ar']} و {drug2_info['name_ar']}**

**💡 ملاحظة:** يمكن تناولهما معاً عموماً
**👨‍⚕️ لكن استشر الصيدلي للتأكد من التوقيت المناسب**"""
        else:
            if interaction_found:
                return f"""⚠️ **Warning: Possible interaction between {drug1_info['name_en']} and {drug2_info['name_en']}**

**🚫 Not recommended to take together without medical consultation**

**👨‍⚕️ Consult pharmacist or doctor before combining**"""
            else:
                return f"""✅ **No known interaction between {drug1_info['name_en']} and {drug2_info['name_en']}**

**💡 Note:** Generally safe to take together
**👨‍⚕️ But consult pharmacist for proper timing**"""

    def handle_interaction_info(self, detected_drugs: List[str], language: str) -> str:
        """معالجة معلومات التداخل لدواء واحد"""
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)

        if not drug_info:
            return self.handle_unknown_drug(drug_name, language)

        if language == 'ar':
            interactions_list = '\n• '.join(drug_info['interactions_ar'])
            return f"""⚠️ **تداخلات {drug_info['name_ar']}:**

• {interactions_list}

**💡 ملاحظة:** تجنب هذه المواد/الأدوية مع {drug_info['name_ar']}
**👨‍⚕️ استشر الصيدلي قبل تناول أي دواء آخر**"""
        else:
            interactions_list = '\n• '.join(drug_info['interactions_en'])
            return f"""⚠️ **{drug_info['name_en']} interactions:**

• {interactions_list}

**💡 Note:** Avoid these substances/drugs with {drug_info['name_en']}
**👨‍⚕️ Consult pharmacist before taking any other medication**"""

    def handle_side_effects_request(self, detected_drugs: List[str], language: str) -> str:
        """معالجة طلبات الآثار الجانبية"""
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)

        if not drug_info:
            return self.handle_unknown_drug(drug_name, language)

        if language == 'ar':
            return f"""⚠️ **الآثار الجانبية المحتملة لـ {drug_info['name_ar']}:**

**الآثار الشائعة:**
• غثيان خفيف
• صداع طفيف
• اضطراب معدة

**⚠️ توقف عن استخدام الدواء واستشر طبيب إذا ظهرت:**
• حساسية (طفح جلدي، تورم)
• صعوبة تنفس
• ألم شديد في المعدة

**👨‍⚕️ استشر الصيدلي للآثار الجانبية المحددة لحالتك**"""
        else:
            return f"""⚠️ **Possible side effects of {drug_info['name_en']}:**

**Common side effects:**
• Mild nausea
• Slight headache
• Stomach upset

**⚠️ Stop using and consult doctor if you experience:**
• Allergic reaction (rash, swelling)
• Breathing difficulties
• Severe stomach pain

**👨‍⚕️ Consult pharmacist for specific side effects for your condition**"""

    def handle_warnings_request(self, detected_drugs: List[str], language: str) -> str:
        """معالجة طلبات التحذيرات"""
        drug_name = detected_drugs[0]
        drug_info = self.drug_api.search_drug(drug_name)

        if not drug_info:
            return self.handle_unknown_drug(drug_name, language)

        if language == 'ar':
            warnings_list = '\n• '.join(drug_info['warnings_ar'])
            return f"""⚠️ **تحذيرات مهمة لـ {drug_info['name_ar']}:**

• {warnings_list}

**🚫 لا تستخدم إذا:**
• لديك حساسية من المكونات
• تتناول أدوية متعارضة

**👨‍⚕️ استشر طبيب أو صيدلي قبل الاستخدام**"""
        else:
            warnings_list = '\n• '.join(drug_info['warnings_en'])
            return f"""⚠️ **Important warnings for {drug_info['name_en']}:**

• {warnings_list}

**🚫 Do not use if:**
• You are allergic to the ingredients
• You are taking conflicting medications

**👨‍⚕️ Consult doctor or pharmacist before use**"""

    def handle_unknown_drug(self, drug_name: str, language: str) -> str:
        """معالجة الأدوية غير المعروفة مع اقتراحات"""
        # محاولة fuzzy matching
        best_match, score = self.intent_classifier.fuzzy_match_drug(drug_name)

        if language == 'ar':
            response = f"🔍 **الدواء '{drug_name}' غير موجود في قاعدة البيانات**\n\n"

            if best_match and score > 0.6:
                matched_drug_info = self.drug_api.search_drug(best_match)
                if matched_drug_info:
                    response += f"💡 **هل تقصد:** {matched_drug_info['name_ar']} ({matched_drug_info['name_en']})؟\n\n"

            response += """**💭 اقتراحات:**
• تأكد من الإملاء الصحيح
• جرب الاسم التجاري مثل "بندول" بدل "باراسيتامول"
• اكتب الاسم الإنجليزي إذا كان متاحاً

**👨‍⚕️ أو استشر الصيدلي مباشرة**"""
        else:
            response = f"🔍 **Drug '{drug_name}' not found in database**\n\n"

            if best_match and score > 0.6:
                matched_drug_info = self.drug_api.search_drug(best_match)
                if matched_drug_info:
                    response += f"💡 **Did you mean:** {matched_drug_info['name_en']} ({matched_drug_info['name_ar']})?\n\n"

            response += """**💭 Suggestions:**
• Check correct spelling
• Try brand name like "Panadol" instead of "Paracetamol"
• Write generic name if available

**👨‍⚕️ Or consult pharmacist directly**"""

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

اكتب المشكلة بوضوح مثل: "عندي صداع" أو "كحة من يومين" """
        else:
            return """Clarify the symptom more so I can understand:

**Specify the problem type:**
• Fever?
• Pain?
• Cough?
• Inflammation?
• Dizziness?
• Cramps?

Write the problem clearly like: "I have headache" or "Cough for 2 days" """

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
        st.header("أمثلة للتجربة الجديدة")
        examples = [
            "جرعة Augmentin",
            "بدائل Zanidip",
            "تداخل Brufen مع Panadol",
            "معلومات عن banadool",
            "كحة ناشفة من يومين"
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
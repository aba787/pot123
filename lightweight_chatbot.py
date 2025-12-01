
import streamlit as st
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import difflib

class LightweightMedicalBot:
    def __init__(self):
        self.load_dataset()
        self.setup_safety_rules()
    
    def load_dataset(self):
        """تحميل قاعدة البيانات من ملف JSON"""
        try:
            with open('medical_dataset_final.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.drug_database = data.get('drug_database', {})
                self.safety_keywords = data.get('safety_keywords', {})
        except FileNotFoundError:
            st.error("ملف قاعدة البيانات غير موجود")
            self.drug_database = {}
            self.safety_keywords = {}
    
    def setup_safety_rules(self):
        """إعداد قوائم السلامة"""
        self.child_keywords = self.safety_keywords.get('children', {})
        self.pregnancy_keywords = self.safety_keywords.get('pregnancy', {})
        self.emergency_keywords = self.safety_keywords.get('emergency', {})
        
        # قائمة أسماء الأدوية التجارية
        self.drug_synonyms = {}
        for drug_key, drug_info in self.drug_database.items():
            brand_names = drug_info.get('brand_names', [])
            for brand in brand_names:
                self.drug_synonyms[brand.lower()] = drug_key
            self.drug_synonyms[drug_info.get('name_ar', '').lower()] = drug_key
            self.drug_synonyms[drug_info.get('name_en', '').lower()] = drug_key
    
    def check_safety_violations(self, user_input: str, language: str) -> Dict:
        """فحص انتهاكات السلامة"""
        user_input_lower = user_input.lower()
        
        # فحص كلمات الأطفال
        child_words = self.child_keywords.get(language, [])
        for word in child_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'child_detected',
                    'message': 'هذه حالة أطفال، استشر الصيدلي مباشرة.' if language == 'ar' else 'Pediatric case, consult pharmacist directly.'
                }
        
        # فحص كلمات الحوامل
        pregnancy_words = self.pregnancy_keywords.get(language, [])
        for word in pregnancy_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'pregnancy_detected',
                    'message': 'الحوامل والمرضعات، استشر الصيدلي مباشرة.' if language == 'ar' else 'Pregnant/nursing women, consult pharmacist directly.'
                }
        
        # فحص كلمات الطوارئ
        emergency_words = self.emergency_keywords.get(language, [])
        for word in emergency_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'emergency_detected',
                    'message': '🚨 هذه علامة خطر. توجه للطوارئ فوراً أو اتصل بـ 997.' if language == 'ar' else '🚨 Emergency sign. Go to emergency or call 997.'
                }
        
        return {'violation': False}
    
    def find_drug(self, text: str) -> Optional[str]:
        """البحث عن دواء في النص"""
        text_lower = text.lower()
        
        # البحث المباشر
        for synonym, drug_key in self.drug_synonyms.items():
            if synonym in text_lower:
                return drug_key
        
        # البحث التقريبي
        words = text_lower.split()
        for word in words:
            if len(word) > 3:
                matches = difflib.get_close_matches(word, self.drug_synonyms.keys(), n=1, cutoff=0.7)
                if matches:
                    return self.drug_synonyms[matches[0]]
        
        return None
    
    def detect_intent(self, user_input: str) -> str:
        """كشف نية المستخدم"""
        text_lower = user_input.lower()
        
        if any(word in text_lower for word in ['جرعة', 'كمية', 'dosage', 'dose']):
            return 'dosage_request'
        elif any(word in text_lower for word in ['بديل', 'بدائل', 'alternative']):
            return 'alternatives_request'
        elif any(word in text_lower for word in ['تداخل', 'تفاعل', 'interaction']):
            return 'interaction_check'
        elif any(word in text_lower for word in ['أعراض جانبية', 'side effects']):
            return 'side_effects'
        elif any(word in text_lower for word in ['تحذير', 'warning']):
            return 'warnings'
        else:
            return 'drug_info'
    
    def process_query(self, user_input: str, language: str) -> str:
        """معالجة الاستفسار"""
        # فحص السلامة أولاً
        safety_check = self.check_safety_violations(user_input, language)
        if safety_check['violation']:
            return safety_check['message']
        
        # البحث عن دواء
        drug_key = self.find_drug(user_input)
        if not drug_key:
            return self.handle_unknown_drug(user_input, language)
        
        drug_info = self.drug_database.get(drug_key)
        if not drug_info:
            return self.handle_unknown_drug(user_input, language)
        
        # تحديد نوع الطلب
        intent = self.detect_intent(user_input)
        
        if intent == 'dosage_request':
            return self.handle_dosage_request(drug_info, language)
        elif intent == 'alternatives_request':
            return self.handle_alternatives(drug_info, language)
        elif intent == 'interaction_check':
            return self.handle_interactions(drug_info, language)
        elif intent == 'side_effects':
            return self.handle_side_effects(drug_info, language)
        elif intent == 'warnings':
            return self.handle_warnings(drug_info, language)
        else:
            return self.handle_drug_info(drug_info, language)
    
    def handle_dosage_request(self, drug_info: Dict, language: str) -> str:
        """رفض إعطاء جرعات"""
        if language == 'ar':
            return f"""🚫 لا يمكنني إعطاء جرعة {drug_info['name_ar']}

⚠️ الجرعة تحتاج حساب دقيق حسب العمر والوزن والحالة الصحية.

👨‍⚕️ استشر صيدلي أو طبيب للجرعة الصحيحة"""
        else:
            return f"""🚫 Cannot provide dosage for {drug_info['name_en']}

⚠️ Dosage requires precise calculation based on age, weight, and condition.

👨‍⚕️ Consult pharmacist or doctor for correct dosage"""
    
    def handle_alternatives(self, drug_info: Dict, language: str) -> str:
        """معالجة البدائل"""
        if language == 'ar':
            alternatives = '\n• '.join(drug_info.get('alternatives_ar', []))
            return f"""💊 بدائل {drug_info['name_ar']}:

• {alternatives}

👨‍⚕️ استشر الصيدلي قبل التبديل"""
        else:
            alternatives = '\n• '.join(drug_info.get('alternatives_en', []))
            return f"""💊 Alternatives to {drug_info['name_en']}:

• {alternatives}

👨‍⚕️ Consult pharmacist before switching"""
    
    def handle_interactions(self, drug_info: Dict, language: str) -> str:
        """معالجة التداخلات"""
        if language == 'ar':
            interactions = '\n• '.join(drug_info.get('interactions_ar', []))
            return f"""⚠️ تداخلات {drug_info['name_ar']}:

• {interactions}

👨‍⚕️ تجنب هذه المواد مع الدواء"""
        else:
            interactions = '\n• '.join(drug_info.get('interactions_en', []))
            return f"""⚠️ {drug_info['name_en']} interactions:

• {interactions}

👨‍⚕️ Avoid these substances with the medication"""
    
    def handle_side_effects(self, drug_info: Dict, language: str) -> str:
        """معالجة الآثار الجانبية"""
        if language == 'ar':
            return f"""⚠️ الآثار الجانبية المحتملة لـ {drug_info['name_ar']}:

• غثيان خفيف
• صداع طفيف
• اضطراب معدة

⚠️ توقف واستشر طبيب إذا ظهرت:
• حساسية أو طفح جلدي
• صعوبة تنفس
• ألم شديد

👨‍⚕️ استشر الصيدلي لمعلومات محددة"""
        else:
            return f"""⚠️ Possible side effects of {drug_info['name_en']}:

• Mild nausea
• Slight headache
• Stomach upset

⚠️ Stop and consult doctor if you experience:
• Allergic reaction or rash
• Breathing difficulties
• Severe pain

👨‍⚕️ Consult pharmacist for specific information"""
    
    def handle_warnings(self, drug_info: Dict, language: str) -> str:
        """معالجة التحذيرات"""
        if language == 'ar':
            warnings = '\n• '.join(drug_info.get('warnings_ar', []))
            return f"""⚠️ تحذيرات مهمة لـ {drug_info['name_ar']}:

• {warnings}

👨‍⚕️ استشر طبيب قبل الاستخدام"""
        else:
            warnings = '\n• '.join(drug_info.get('warnings_en', []))
            return f"""⚠️ Important warnings for {drug_info['name_en']}:

• {warnings}

👨‍⚕️ Consult doctor before use"""
    
    def handle_drug_info(self, drug_info: Dict, language: str) -> str:
        """معلومات عامة عن الدواء"""
        if language == 'ar':
            return f"""💊 {drug_info['name_ar']} ({drug_info['name_en']})

🔹 الاستخدام: {drug_info['general_use_ar']}
🔹 التحذيرات: {', '.join(drug_info.get('warnings_ar', [])[:2])}
🔹 التداخلات: {', '.join(drug_info.get('interactions_ar', [])[:2])}

⚠️ بدون جرعة - استشر الصيدلي"""
        else:
            return f"""💊 {drug_info['name_en']} ({drug_info['name_ar']})

🔹 Use: {drug_info['general_use_en']}
🔹 Warnings: {', '.join(drug_info.get('warnings_en', [])[:2])}
🔹 Interactions: {', '.join(drug_info.get('interactions_en', [])[:2])}

⚠️ No dosage - consult pharmacist"""
    
    def handle_unknown_drug(self, drug_name: str, language: str) -> str:
        """معالجة الأدوية غير المعروفة"""
        if language == 'ar':
            return f"""🔍 الدواء '{drug_name}' غير موجود في قاعدة البيانات

💭 اقتراحات:
• تأكد من الإملاء الصحيح
• جرب الاسم التجاري
• استشر الصيدلي مباشرة"""
        else:
            return f"""🔍 Drug '{drug_name}' not found in database

💭 Suggestions:
• Check correct spelling
• Try brand name
• Consult pharmacist directly"""
    
    def detect_language(self, text: str) -> str:
        """كشف لغة النص"""
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        return 'ar' if len(arabic_chars) > len(text) * 0.3 else 'en'

def main():
    st.set_page_config(
        page_title="البوت الطبي الآمن - النسخة الخفيفة",
        page_icon="💊",
        layout="wide"
    )
    
    st.title("💊 البوت الطبي الآمن - النسخة الخفيفة")
    st.markdown("### Safe Medical Bot - Lightweight Version")
    
    # تهيئة البوت
    if 'bot' not in st.session_state:
        st.session_state.bot = LightweightMedicalBot()
        st.session_state.chat_history = []
    
    # الشريط الجانبي
    with st.sidebar:
        st.header("قواعد السلامة")
        st.success("✅ منع جرعات الأطفال")
        st.success("✅ منع وصف للحوامل") 
        st.success("✅ تحويل الطوارئ")
        st.success("✅ بدون جرعات نهائياً")
        
        if st.button("مسح المحادثة"):
            st.session_state.chat_history = []
            st.rerun()
    
    # عرض تاريخ المحادثة
    if st.session_state.chat_history:
        for i, (user_msg, bot_response, timestamp) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**أنت ({timestamp}):** {user_msg}")
                st.markdown(f"**البوت:** {bot_response}")
                if i < len(st.session_state.chat_history) - 1:
                    st.markdown("---")
    
    # إدخال الرسالة
    user_input = st.text_area(
        "اكتب رسالتك:",
        placeholder="مثال: معلومات عن بندول، أو بدائل أوجمنتين"
    )
    
    if st.button("إرسال", type="primary"):
        if user_input:
            language = st.session_state.bot.detect_language(user_input)
            response = st.session_state.bot.process_query(user_input, language)
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            st.session_state.chat_history.append((user_input, response, timestamp))
            st.rerun()

if __name__ == "__main__":
    main()

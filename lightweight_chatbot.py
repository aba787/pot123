
import streamlit as st
import json
import re
from datetime import datetime
from typing import Dict, List, Optional
import difflib
import os

class LightweightMedicalBot:
    def __init__(self):
        self.load_dataset()
        self.setup_safety_rules()
    
    def load_dataset(self):
        """تحميل قاعدة البيانات من ملف JSON"""
        try:
            if not os.path.exists('medical_dataset_final.json'):
                st.error("❌ ملف قاعدة البيانات غير موجود: medical_dataset_final.json")
                self.drug_database = {}
                self.safety_keywords = {}
                return
                
            with open('medical_dataset_final.json', 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.drug_database = data.get('drug_database', {})
                self.safety_keywords = data.get('safety_keywords', {})
                
        except Exception as e:
            st.error(f"❌ خطأ في تحميل قاعدة البيانات: {str(e)}")
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
                    'message': '🚫 هذه حالة أطفال، استشر الصيدلي مباشرة.' if language == 'ar' else '🚫 Pediatric case, consult pharmacist directly.'
                }
        
        # فحص كلمات الحوامل
        pregnancy_words = self.pregnancy_keywords.get(language, [])
        for word in pregnancy_words:
            if word in user_input_lower:
                return {
                    'violation': True,
                    'type': 'pregnancy_detected',
                    'message': '🚫 الحوامل والمرضعات، استشر الصيدلي مباشرة.' if language == 'ar' else '🚫 Pregnant/nursing women, consult pharmacist directly.'
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
    
    def normalize_arabic_text(self, text: str) -> str:
        """تطبيع النص العربي"""
        text = text.lower()
        # إزالة الهمزات
        text = text.replace('أ', 'ا').replace('إ', 'ا').replace('آ', 'ا')
        text = text.replace('ى', 'ي').replace('ة', 'ه')
        return text.strip()
    
    def smart_search(self, query: str) -> Optional[str]:
        """البحث الذكي في قاعدة البيانات"""
        query_normalized = self.normalize_arabic_text(query)
        query_lower = query.lower()
        
        # 1. البحث المباشر في أسماء الأدوية والأسماء التجارية
        for synonym, drug_key in self.drug_synonyms.items():
            if synonym in query_lower or synonym in query_normalized:
                return drug_key
        
        # 2. البحث في الاستخدامات (general_use)
        for drug_key, drug_info in self.drug_database.items():
            # البحث في الاستخدام العربي
            use_ar = drug_info.get('general_use_ar', '').lower()
            use_ar_normalized = self.normalize_arabic_text(use_ar)
            
            # البحث في الاستخدام الإنجليزي
            use_en = drug_info.get('general_use_en', '').lower()
            
            # فحص الكلمات المفتاحية
            search_terms = [
                # كلمات الصداع
                'صداع', 'headache', 'رأس', 'head',
                # كلمات الألم
                'ألم', 'pain', 'وجع', 'ache',
                # كلمات الحرارة
                'حرارة', 'fever', 'سخونة', 'temperature',
                # كلمات المضاد الحيوي
                'التهاب', 'infection', 'بكتيريا', 'bacterial',
                # كلمات عامة
                'مسكن', 'painkiller', 'خافض', 'reducer'
            ]
            
            for term in search_terms:
                term_normalized = self.normalize_arabic_text(term)
                if ((term in query_lower or term_normalized in query_normalized) and
                    (term in use_ar or term_normalized in use_ar_normalized or term in use_en)):
                    return drug_key
        
        # 3. البحث التقريبي في أسماء الأدوية
        words = query_lower.split()
        for word in words:
            if len(word) > 3:
                matches = difflib.get_close_matches(word, self.drug_synonyms.keys(), n=1, cutoff=0.7)
                if matches:
                    return self.drug_synonyms[matches[0]]
        
        return None
    
    def find_drug(self, text: str) -> Optional[str]:
        """البحث عن دواء في النص باستخدام البحث الذكي"""
        return self.smart_search(text)
    
    def detect_intent_filter(self, query: str) -> str:
        """فلتر النوايا قبل البحث الطبي"""
        greetings = ["مرحبا", "هلا", "السلام عليكم", "hello", "hi", "hey", "أهلا", "سلام", "هلو"]
        smalltalk = ["كيفك", "شلونك", "كيف الحال", "وش الاخبار", "how are you", "what's up", "كيف حالك"]
        
        q = query.strip().lower()
        
        # تحيات
        for g in greetings:
            if g in q:
                return "greeting"
        
        # كلام عام
        for s in smalltalk:
            if s in q:
                return "smalltalk"
        
        return "medical"
    
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
    
    def detect_language(self, text: str) -> str:
        """كشف لغة النص"""
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        return 'ar' if len(arabic_chars) > len(text) * 0.3 else 'en'
    
    def process_user_input(self, user_input: str) -> str:
        """معالجة مدخل المستخدم وإرجاع الرد"""
        if not user_input or not user_input.strip():
            return "يرجى كتابة سؤالك أولاً"
        
        # كشف اللغة
        language = self.detect_language(user_input)
        
        # فحص السلامة أولاً
        safety_check = self.check_safety_violations(user_input, language)
        if safety_check['violation']:
            return safety_check['message']
        
        # فلتر النوايا قبل البحث الطبي
        intent_filter = self.detect_intent_filter(user_input)
        
        if intent_filter == "greeting":
            if language == 'ar':
                return "أهلاً وسهلاً! 💊 كيف أقدر أساعدك طبياً اليوم؟"
            else:
                return "Hello! 💊 How can I help you medically today?"
        
        if intent_filter == "smalltalk":
            if language == 'ar':
                return "تمام الحمد لله! 😊 كيف أقدر أساعدك طبياً؟"
            else:
                return "I'm doing well, thank you! 😊 How can I help you medically?"
        
        # إذا كانت النية طبية، نتابع البحث
        if intent_filter == "medical":
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
        
        # إذا لم نتمكن من تحديد النية
        if language == 'ar':
            return "عذراً، لم أتمكن من فهم طلبك. يرجى كتابة سؤال طبي واضح أو اسم دواء."
        else:
            return "Sorry, I couldn't understand your request. Please write a clear medical question or drug name."
    
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
            alternatives = drug_info.get('alternatives_ar', [])
            alternatives_text = '\n• '.join(alternatives) if alternatives else "لا توجد بدائل مسجلة"
            return f"""💊 بدائل {drug_info['name_ar']}:

• {alternatives_text}

👨‍⚕️ استشر الصيدلي قبل التبديل"""
        else:
            alternatives = drug_info.get('alternatives_en', [])
            alternatives_text = '\n• '.join(alternatives) if alternatives else "No alternatives recorded"
            return f"""💊 Alternatives to {drug_info['name_en']}:

• {alternatives_text}

👨‍⚕️ Consult pharmacist before switching"""
    
    def handle_interactions(self, drug_info: Dict, language: str) -> str:
        """معالجة التداخلات"""
        if language == 'ar':
            interactions = drug_info.get('interactions_ar', [])
            interactions_text = '\n• '.join(interactions) if interactions else "لا توجد تداخلات مسجلة"
            return f"""⚠️ تداخلات {drug_info['name_ar']}:

• {interactions_text}

👨‍⚕️ تجنب هذه المواد مع الدواء"""
        else:
            interactions = drug_info.get('interactions_en', [])
            interactions_text = '\n• '.join(interactions) if interactions else "No interactions recorded"
            return f"""⚠️ {drug_info['name_en']} interactions:

• {interactions_text}

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
            warnings = drug_info.get('warnings_ar', [])
            warnings_text = '\n• '.join(warnings) if warnings else "لا توجد تحذيرات مسجلة"
            return f"""⚠️ تحذيرات مهمة لـ {drug_info['name_ar']}:

• {warnings_text}

👨‍⚕️ استشر طبيب قبل الاستخدام"""
        else:
            warnings = drug_info.get('warnings_en', [])
            warnings_text = '\n• '.join(warnings) if warnings else "No warnings recorded"
            return f"""⚠️ Important warnings for {drug_info['name_en']}:

• {warnings_text}

👨‍⚕️ Consult doctor before use"""
    
    def handle_drug_info(self, drug_info: Dict, language: str) -> str:
        """معلومات عامة عن الدواء"""
        if language == 'ar':
            return f"""💊 {drug_info['name_ar']} ({drug_info['name_en']})

🔹 الاستخدام: {drug_info.get('general_use_ar', 'غير محدد')}
🔹 التحذيرات: {', '.join(drug_info.get('warnings_ar', ['لا توجد'])[:2])}
🔹 التداخلات: {', '.join(drug_info.get('interactions_ar', ['لا توجد'])[:2])}

⚠️ بدون جرعة - استشر الصيدلي"""
        else:
            return f"""💊 {drug_info['name_en']} ({drug_info['name_ar']})

🔹 Use: {drug_info.get('general_use_en', 'Not specified')}
🔹 Warnings: {', '.join(drug_info.get('warnings_en', ['None'])[:2])}
🔹 Interactions: {', '.join(drug_info.get('interactions_en', ['None'])[:2])}

⚠️ No dosage - consult pharmacist"""
    
    def handle_unknown_drug(self, query: str, language: str) -> str:
        """معالجة الاستفسارات غير المعروفة مع اقتراحات ذكية"""
        
        # محاولة تقديم اقتراحات حسب الكلمات المفتاحية
        query_lower = query.lower()
        query_normalized = self.normalize_arabic_text(query)
        
        suggestions = []
        
        # اقتراحات حسب الأعراض الشائعة
        if any(term in query_lower or term in query_normalized for term in ['صداع', 'headache', 'رأس']):
            suggestions.append("باراسيتامول (بندول) للصداع")
        
        if any(term in query_lower or term in query_normalized for term in ['حرارة', 'fever', 'سخونة']):
            suggestions.append("باراسيتامول (بندول) لخفض الحرارة")
        
        if any(term in query_lower or term in query_normalized for term in ['التهاب', 'infection', 'بكتيريا']):
            suggestions.append("أوجمنتين للالتهابات البكتيرية")
        
        if language == 'ar':
            suggestions_text = '\n• '.join(suggestions) if suggestions else "لا توجد اقتراحات محددة"
            return f"""🔍 لم أجد معلومات محددة عن "{query}"

💡 اقتراحات قد تفيدك:
• {suggestions_text}

💭 نصائح للبحث:
• اكتب اسم الدواء بوضوح (مثل: بندول، أوجمنتين)
• أو اكتب العرض (مثل: دواء للصداع، دواء للحرارة)
• استشر الصيدلي للحصول على المشورة المناسبة

💊 أدوية متاحة في قاعدة البيانات: باراسيتامول، بندول، أوجمنتين"""
        else:
            suggestions_text = '\n• '.join(suggestions) if suggestions else "No specific suggestions available"
            return f"""🔍 Could not find specific information about "{query}"

💡 Suggestions that might help:
• {suggestions_text}

💭 Search tips:
• Write drug name clearly (e.g: Panadol, Augmentin)
• Or write the symptom (e.g: medicine for headache, fever reducer)
• Consult pharmacist for appropriate advice

💊 Available drugs in database: Paracetamol, Panadol, Augmentin"""

def process_user_input(user_text):
    """دالة معالجة النص الرئيسية"""
    if 'bot' not in st.session_state:
        st.session_state.bot = LightweightMedicalBot()
    
    return st.session_state.bot.process_user_input(user_text)

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
        st.success("✅ تم تحميل البوت بنجاح!")
    
    # عرض معلومات حالة النظام
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد الأدوية", len(st.session_state.bot.drug_database))
    with col2:
        st.metric("حالة قاعدة البيانات", "✅ متصل" if st.session_state.bot.drug_database else "❌ غير متصل")
    with col3:
        st.metric("نظام الأمان", "✅ فعال")
    
    st.markdown("---")
    
    # واجهة الإدخال المطلوبة
    user_input = st.text_input("اكتب سؤالك:")
    
    if user_input:
        with st.spinner("جاري المعالجة..."):
            response = process_user_input(user_input)
            st.write(response)
    
    # أمثلة للاستخدام
    st.markdown("### 💡 أمثلة للتجربة:")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("معلومات عن بندول"):
            response = process_user_input("معلومات عن بندول")
            st.write(response)
        
        if st.button("بدائل أوجمنتين"):
            response = process_user_input("بدائل أوجمنتين")
            st.write(response)
    
    with col2:
        if st.button("تداخل الأدوية"):
            response = process_user_input("تداخل باراسيتامول")
            st.write(response)
        
        if st.button("Information about Paracetamol"):
            response = process_user_input("Information about Paracetamol")
            st.write(response)
    
    # تحذيرات الأمان
    with st.expander("🚫 أمثلة محظورة - سيرفضها النظام"):
        st.error("جرعة بندول للطفل - سيحول للصيدلي")
        st.error("دواء آمن للحامل - سيحول للصيدلي") 
        st.error("عندي ألم في الصدر - سيحول للطوارئ")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "🏥 البوت الطبي الآمن | لأغراض تعليمية | لا يغني عن الاستشارة الطبية"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()


import requests
import json
import os
from typing import Dict, List, Optional, Any
import openai
from datetime import datetime

class MedicalAPIHandler:
    def __init__(self):
        self.setup_apis()
        
    def setup_apis(self):
        """إعداد الـ APIs الطبية"""
        # OpenFDA API - مجاني ولا يحتاج API key
        self.openfda_base_url = "https://api.fda.gov/drug"
        
        # OpenAI API - يحتاج API key
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        if self.openai_api_key:
            openai.api_key = self.openai_api_key
            
        # NHS API - مجاني
        self.nhs_base_url = "https://api.nhs.uk/medicines"
        
        # DrugBank API - يحتاج API key (اختياري)
        self.drugbank_api_key = os.getenv('DRUGBANK_API_KEY')
        
    def search_openfda(self, drug_name: str) -> Optional[Dict]:
        """البحث في OpenFDA API"""
        try:
            # البحث في قاعدة بيانات الأدوية المعتمدة
            url = f"{self.openfda_base_url}/label.json"
            params = {
                'search': f'openfda.brand_name:"{drug_name}" OR openfda.generic_name:"{drug_name}"',
                'limit': 1
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'results' in data and len(data['results']) > 0:
                    result = data['results'][0]
                    return self.parse_fda_data(result)
                    
        except Exception as e:
            print(f"OpenFDA API error: {str(e)}")
            
        return None
    
    def parse_fda_data(self, fda_result: Dict) -> Dict:
        """تحليل بيانات FDA وتنظيمها"""
        try:
            openfda = fda_result.get('openfda', {})
            
            parsed_data = {
                'name': openfda.get('brand_name', ['Unknown'])[0] if openfda.get('brand_name') else 'Unknown',
                'generic_name': openfda.get('generic_name', ['Unknown'])[0] if openfda.get('generic_name') else 'Unknown',
                'manufacturer': openfda.get('manufacturer_name', ['Unknown'])[0] if openfda.get('manufacturer_name') else 'Unknown',
                'indications': fda_result.get('indications_and_usage', ['Not specified'])[0][:500] if fda_result.get('indications_and_usage') else 'Not specified',
                'warnings': fda_result.get('warnings', ['Not specified'])[0][:500] if fda_result.get('warnings') else 'Not specified',
                'dosage': fda_result.get('dosage_and_administration', ['Consult healthcare provider'])[0][:300] if fda_result.get('dosage_and_administration') else 'Consult healthcare provider',
                'contraindications': fda_result.get('contraindications', ['Not specified'])[0][:300] if fda_result.get('contraindications') else 'Not specified',
                'source': 'FDA'
            }
            
            return parsed_data
            
        except Exception as e:
            print(f"Error parsing FDA data: {str(e)}")
            return None
    
    def search_medical_apis(self, query: str) -> Optional[Dict]:
        """البحث في جميع الـ APIs الطبية المتاحة"""
        
        # تنظيف الاستعلام
        clean_query = self.clean_medical_query(query)
        
        # البحث في OpenFDA
        fda_result = self.search_openfda(clean_query)
        if fda_result:
            return fda_result
            
        # يمكن إضافة APIs أخرى هنا
        # nhs_result = self.search_nhs(clean_query)
        # drugbank_result = self.search_drugbank(clean_query)
        
        return None
    
    def clean_medical_query(self, query: str) -> str:
        """تنظيف الاستعلام الطبي"""
        # إزالة الكلمات غير المفيدة
        stop_words = [
            'دواء', 'medicine', 'medication', 'للـ', 'for', 'عن', 'about',
            'معلومات', 'information', 'ما هو', 'what is', 'كيف', 'how'
        ]
        
        cleaned = query.lower()
        for word in stop_words:
            cleaned = cleaned.replace(word, ' ')
            
        # إزالة المسافات الزائدة
        cleaned = ' '.join(cleaned.split())
        
        return cleaned.strip()
    
    def ask_ai_model(self, query: str, language: str = 'ar') -> Optional[str]:
        """استخدام نموذج AI كبديل"""
        
        if not self.openai_api_key:
            return self.get_fallback_ai_response(query, language)
        
        try:
            # إعداد الـ prompt للمساعد الطبي
            system_prompt = """أنت مساعد طبي تعليمي. قدم معلومات عامة فقط.
            
قواعد مهمة:
1. لا تقدم تشخيص طبي
2. لا تقدم جرعات محددة  
3. أكد دائماً على استشارة الطبيب
4. قدم معلومات عامة تعليمية فقط
5. إذا كان السؤال عن أطفال أو حوامل، أحل فوراً للطبيب
6. إذا كانت أعراض طوارئ، أحل للطوارئ فوراً

ابدأ كل إجابة بـ: "هذه معلومات عامة تعليمية فقط"
اختتم كل إجابة بـ: "استشر طبيبك للحصول على المشورة الطبية المناسبة" """

            if language == 'en':
                system_prompt = """You are an educational medical assistant. Provide general information only.

Important rules:
1. Do not provide medical diagnosis
2. Do not provide specific dosages
3. Always emphasize consulting a doctor
4. Provide general educational information only
5. If question is about children or pregnant women, refer immediately to doctor
6. If emergency symptoms, refer to emergency immediately

Start every answer with: "This is general educational information only"
End every answer with: "Consult your doctor for appropriate medical advice" """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API error: {str(e)}")
            return self.get_fallback_ai_response(query, language)
    
    def get_fallback_ai_response(self, query: str, language: str) -> str:
        """رد بديل عند فشل AI APIs"""
        
        # تحليل بسيط للكلمات المفتاحية
        query_lower = query.lower()
        
        # ردود جاهزة للأعراض الشائعة
        symptom_responses = {
            'ar': {
                'صداع': """🔍 هذه معلومات عامة تعليمية فقط

💊 **الصداع العام:**
• يمكن أن يكون بسبب التوتر، قلة النوم، أو الجفاف
• المسكنات البسيطة قد تساعد (مثل الباراسيتامول)
• الراحة وشرب الماء مهم

⚠️ **راجع الطبيب إذا:**
• الصداع شديد ومفاجئ
• مصحوب بحمى أو تيبس الرقبة
• يزداد سوءاً مع الوقت

**استشر طبيبك للحصول على المشورة الطبية المناسبة**""",

                'سعال': """🔍 هذه معلومات عامة تعليمية فقط

💊 **السعال العام:**
• قد يكون بسبب التهاب الجهاز التنفسي العلوي
• السوائل الدافئة والعسل قد تساعد
• تجنب المهيجات مثل الدخان

⚠️ **راجع الطبيب إذا:**
• السعال مستمر أكثر من أسبوعين
• مصحوب بدم أو حمى عالية
• صعوبة في التنفس

**استشر طبيبك للحصول على المشورة الطبية المناسبة**""",

                'حرارة': """🔍 هذه معلومات عامة تعليمية فقط

💊 **الحمى العامة:**
• علامة على أن الجسم يحارب العدوى
• الراحة وشرب السوائل مهم
• خافضات الحرارة قد تساعد في الراحة

⚠️ **راجع الطبيب إذا:**
• الحرارة أعلى من 39 درجة
• مستمرة أكثر من 3 أيام
• مصحوبة بأعراض خطيرة

**استشر طبيبك للحصول على المشورة الطبية المناسبة**"""
            },
            'en': {
                'headache': """🔍 This is general educational information only

💊 **General Headache:**
• Can be caused by stress, lack of sleep, or dehydration  
• Simple pain relievers may help (like paracetamol)
• Rest and drinking water is important

⚠️ **See doctor if:**
• Headache is severe and sudden
• Accompanied by fever or neck stiffness
• Gets worse over time

**Consult your doctor for appropriate medical advice**""",

                'cough': """🔍 This is general educational information only

💊 **General Cough:**
• May be due to upper respiratory tract inflammation
• Warm fluids and honey may help
• Avoid irritants like smoke

⚠️ **See doctor if:**
• Cough persists more than two weeks
• Accompanied by blood or high fever
• Difficulty breathing

**Consult your doctor for appropriate medical advice**""",

                'fever': """🔍 This is general educational information only

💊 **General Fever:**
• Sign that body is fighting infection
• Rest and fluid intake is important
• Fever reducers may help comfort

⚠️ **See doctor if:**
• Temperature above 39°C
• Persists more than 3 days  
• Accompanied by serious symptoms

**Consult your doctor for appropriate medical advice**"""
            }
        }
        
        # البحث عن أعراض مطابقة
        responses = symptom_responses.get(language, symptom_responses['ar'])
        
        for symptom, response in responses.items():
            if symptom in query_lower:
                return response
        
        # رد عام إذا لم يجد شيء محدد
        if language == 'ar':
            return """🔍 هذه معلومات عامة تعليمية فقط

💭 **لم أتمكن من العثور على معلومات محددة لاستفسارك**

💡 **نصائح عامة:**
• تأكد من وضوح السؤال
• حدد العرض أو اسم الدواء بدقة
• استشر صيدلي أو طبيب للحصول على إجابة دقيقة

⚠️ **للحالات الطارئة:** اتصل بـ 997 أو توجه لأقرب مستشفى

**استشر طبيبك للحصول على المشورة الطبية المناسبة**"""
        else:
            return """🔍 This is general educational information only

💭 **Could not find specific information for your query**

💡 **General tips:**
• Make sure question is clear
• Specify symptom or drug name accurately  
• Consult pharmacist or doctor for accurate answer

⚠️ **For emergencies:** Call emergency services or go to nearest hospital

**Consult your doctor for appropriate medical advice**"""

class EnhancedMedicalBot:
    def __init__(self):
        self.api_handler = MedicalAPIHandler()
        self.medical_disclaimer = {
            'ar': "\n\n⚠️ **تنبيه طبي:** المعلومات المقدمة هنا لأغراض تعليمية عامة فقط ولا تغني عن الاستشارة الطبية المتخصصة. استشر طبيبك أو صيدلي مختص عند الحاجة.",
            'en': "\n\n⚠️ **Medical Disclaimer:** The information provided here is for general educational purposes only and does not replace professional medical consultation. Consult your doctor or qualified pharmacist when needed."
        }
    
    def process_medical_query(self, query: str, language: str = 'ar') -> str:
        """معالجة الاستفسار الطبي مع API ثم AI كبديل"""
        
        # المنطق المطلوب:
        # 1. البحث في Medical APIs أولاً
        api_result = self.api_handler.search_medical_apis(query)
        
        if api_result:
            # تنسيق نتيجة API
            formatted_response = self.format_api_response(api_result, language)
            return formatted_response + self.medical_disclaimer[language]
        
        # 2. إذا لم تجد API، استخدم AI Model
        ai_response = self.api_handler.ask_ai_model(query, language)
        
        if ai_response:
            return ai_response + self.medical_disclaimer[language]
        
        # 3. رد أساسي إذا فشل كل شيء (لا نقول "لم أجد معلومات" أبداً)
        return self.get_basic_medical_guidance(query, language)
    
    def format_api_response(self, api_data: Dict, language: str) -> str:
        """تنسيق رد API بشكل مفهوم"""
        
        if language == 'ar':
            response = f"""💊 **معلومات من قاعدة البيانات الطبية الرسمية**

🔹 **الاسم:** {api_data.get('name', 'غير محدد')}
🔹 **الاسم العلمي:** {api_data.get('generic_name', 'غير محدد')}
🔹 **الشركة المصنعة:** {api_data.get('manufacturer', 'غير محدد')}

📋 **دواعي الاستعمال:**
{api_data.get('indications', 'استشر الطبيب')}

⚠️ **تحذيرات مهمة:**
{api_data.get('warnings', 'استشر الطبيب')}

🚫 **موانع الاستعمال:**
{api_data.get('contraindications', 'استشر الطبيب')}

💊 **معلومات الجرعة:**
{api_data.get('dosage', 'استشر الطبيب أو الصيدلي للجرعة المناسبة')}

📍 **المصدر:** {api_data.get('source', 'قاعدة بيانات طبية')}"""

        else:
            response = f"""💊 **Information from Official Medical Database**

🔹 **Name:** {api_data.get('name', 'Not specified')}
🔹 **Generic Name:** {api_data.get('generic_name', 'Not specified')}
🔹 **Manufacturer:** {api_data.get('manufacturer', 'Not specified')}

📋 **Indications:**
{api_data.get('indications', 'Consult doctor')}

⚠️ **Important Warnings:**
{api_data.get('warnings', 'Consult doctor')}

🚫 **Contraindications:**
{api_data.get('contraindications', 'Consult doctor')}

💊 **Dosage Information:**
{api_data.get('dosage', 'Consult doctor or pharmacist for appropriate dosage')}

📍 **Source:** {api_data.get('source', 'Medical Database')}"""

        return response
    
    def get_basic_medical_guidance(self, query: str, language: str) -> str:
        """توجيه طبي أساسي عندما تفشل جميع الطرق"""
        
        if language == 'ar':
            return """🏥 **توجيه طبي عام**

💭 **استفسارك يحتاج مشورة طبية متخصصة**

💡 **الخطوات المقترحة:**
1. **استشر صيدلي:** للأدوية والمستحضرات العامة
2. **استشر طبيب:** للأعراض والحالات الطبية
3. **اتصل بـ 997:** للحالات الطارئة

🔍 **نصائح للبحث:**
• اكتب اسم الدواء بوضوح
• حدد العرض بدقة
• اذكر أي معلومات إضافية مهمة

⚕️ **مراكز المساعدة:**
• الصيدليات المحلية
• المراكز الصحية  
• المستشفيات العامة"""

        else:
            return """🏥 **General Medical Guidance**

💭 **Your inquiry needs specialized medical consultation**

💡 **Suggested steps:**
1. **Consult pharmacist:** For medications and general products
2. **Consult doctor:** For symptoms and medical conditions  
3. **Call emergency:** For urgent situations

🔍 **Search tips:**
• Write drug name clearly
• Specify symptom accurately
• Mention any important additional information

⚕️ **Help centers:**
• Local pharmacies
• Health centers
• General hospitals"""
    
    def detect_language(self, text: str) -> str:
        """كشف لغة النص"""
        import re
        arabic_chars = re.findall(r'[\u0600-\u06FF]', text)
        return 'ar' if len(arabic_chars) > len(text) * 0.3 else 'en'

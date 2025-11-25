
import pandas as pd
import json
import random
from datetime import datetime

class MedicalDatasetBuilder:
    def __init__(self):
        self.setup_sample_data()
    
    def setup_sample_data(self):
        """Create sample bilingual medical dataset"""
        self.training_data = [
            # Symptom Inquiry - English
            {"text": "I have a headache", "intent": "symptom_inquiry", "language": "en"},
            {"text": "I feel sick and have fever", "intent": "symptom_inquiry", "language": "en"},
            {"text": "My stomach hurts", "intent": "symptom_inquiry", "language": "en"},
            {"text": "I have been coughing for days", "intent": "symptom_inquiry", "language": "en"},
            {"text": "I feel dizzy and nauseous", "intent": "symptom_inquiry", "language": "en"},
            
            # Symptom Inquiry - Arabic
            {"text": "أعاني من صداع شديد", "intent": "symptom_inquiry", "language": "ar"},
            {"text": "أشعر بألم في معدتي", "intent": "symptom_inquiry", "language": "ar"},
            {"text": "عندي حمى وسعال", "intent": "symptom_inquiry", "language": "ar"},
            {"text": "أشعر بدوخة ووهن", "intent": "symptom_inquiry", "language": "ar"},
            {"text": "يؤلمني ظهري كثيراً", "intent": "symptom_inquiry", "language": "ar"},
            
            # Medication Info - English
            {"text": "What is the dose for paracetamol?", "intent": "medication_info", "language": "en"},
            {"text": "Can I take this medicine with food?", "intent": "medication_info", "language": "en"},
            {"text": "What are the side effects?", "intent": "medication_info", "language": "en"},
            {"text": "How often should I take this pill?", "intent": "medication_info", "language": "en"},
            
            # Medication Info - Arabic
            {"text": "ما هي الجرعة المناسبة؟", "intent": "medication_info", "language": "ar"},
            {"text": "هل يمكن تناول هذا الدواء مع الطعام؟", "intent": "medication_info", "language": "ar"},
            {"text": "ما هي الأعراض الجانبية؟", "intent": "medication_info", "language": "ar"},
            {"text": "كم مرة يجب أن آخذ هذا الدواء؟", "intent": "medication_info", "language": "ar"},
            
            # Appointment - English
            {"text": "I want to book an appointment", "intent": "appointment", "language": "en"},
            {"text": "Can I schedule a visit with the doctor?", "intent": "appointment", "language": "en"},
            {"text": "What are the available appointment slots?", "intent": "appointment", "language": "en"},
            
            # Appointment - Arabic
            {"text": "أريد حجز موعد", "intent": "appointment", "language": "ar"},
            {"text": "هل يمكنني حجز موعد مع الطبيب؟", "intent": "appointment", "language": "ar"},
            {"text": "ما هي المواعيد المتاحة؟", "intent": "appointment", "language": "ar"},
            
            # Image Analysis - English
            {"text": "Can you analyze this X-ray image?", "intent": "image_analysis", "language": "en"},
            {"text": "Please look at this medical scan", "intent": "image_analysis", "language": "en"},
            {"text": "What do you see in this photo?", "intent": "image_analysis", "language": "en"},
            
            # Image Analysis - Arabic  
            {"text": "هل يمكنك تحليل صورة الأشعة هذه؟", "intent": "image_analysis", "language": "ar"},
            {"text": "أرجو النظر في هذا الفحص الطبي", "intent": "image_analysis", "language": "ar"},
            {"text": "ماذا ترى في هذه الصورة؟", "intent": "image_analysis", "language": "ar"},
            
            # Greetings - English
            {"text": "Hello", "intent": "greeting", "language": "en"},
            {"text": "Hi there", "intent": "greeting", "language": "en"},
            {"text": "Good morning", "intent": "greeting", "language": "en"},
            
            # Greetings - Arabic
            {"text": "مرحباً", "intent": "greeting", "language": "ar"},
            {"text": "أهلاً وسهلاً", "intent": "greeting", "language": "ar"},
            {"text": "السلام عليكم", "intent": "greeting", "language": "ar"},
        ]
    
    def save_dataset(self, filename="medical_chatbot_dataset.json"):
        """Save dataset to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {filename}")
        return filename
    
    def create_csv_dataset(self, filename="medical_dataset.csv"):
        """Create CSV dataset for easy analysis"""
        df = pd.DataFrame(self.training_data)
        df.to_csv(filename, index=False, encoding='utf-8')
        print(f"CSV dataset saved to {filename}")
        return filename
    
    def get_statistics(self):
        """Get dataset statistics"""
        df = pd.DataFrame(self.training_data)
        stats = {
            "total_samples": len(df),
            "intents": df['intent'].value_counts().to_dict(),
            "languages": df['language'].value_counts().to_dict(),
            "intent_by_language": df.groupby(['intent', 'language']).size().to_dict()
        }
        return stats

if __name__ == "__main__":
    # Create dataset
    builder = MedicalDatasetBuilder()
    
    # Save files
    builder.save_dataset()
    builder.create_csv_dataset()
    
    # Print statistics
    stats = builder.get_statistics()
    print("\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Intents: {stats['intents']}")
    print(f"Languages: {stats['languages']}")

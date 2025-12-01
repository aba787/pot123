
#!/usr/bin/env python3
"""
ملف تشغيل مبسط للنسخة الخفيفة من البوت الطبي
يمكن استخدامه على أي استضافة خفيفة
"""

import subprocess
import sys
import os

def install_requirements():
    """تثبيت المتطلبات البسيطة"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
        print("✅ تم تثبيت المتطلبات بنجاح")
    except subprocess.CalledProcessError as e:
        print(f"❌ خطأ في تثبيت المتطلبات: {e}")
        return False
    return True

def run_app():
    """تشغيل التطبيق"""
    try:
        # التأكد من وجود الملفات المطلوبة
        required_files = ['lightweight_chatbot.py', 'medical_dataset_final.json']
        for file in required_files:
            if not os.path.exists(file):
                print(f"❌ الملف المطلوب غير موجود: {file}")
                return False
        
        print("🚀 بدء تشغيل البوت الطبي الخفيف...")
        
        # تشغيل streamlit
        subprocess.run([
            "streamlit", "run", "lightweight_chatbot.py",
            "--server.address=0.0.0.0",
            "--server.port=5000",
            "--server.headless=true",
            "--server.enableCORS=false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف التطبيق بواسطة المستخدم")
    except Exception as e:
        print(f"❌ خطأ في تشغيل التطبيق: {e}")

if __name__ == "__main__":
    print("💊 البوت الطبي الآمن - النسخة الخفيفة")
    print("="*50)
    
    # تثبيت المتطلبات
    if install_requirements():
        # تشغيل التطبيق
        run_app()
    else:
        print("❌ فشل في التهيئة")
        sys.exit(1)

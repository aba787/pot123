
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from torch.utils.data import Dataset, DataLoader
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import numpy as np

class MedicalIntentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class MBERTTrainer:
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.intent_to_id = {}
        self.id_to_intent = {}
        
    def prepare_data(self, json_file):
        """Load and prepare training data"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        df = pd.DataFrame(data)
        
        # Create intent mappings
        unique_intents = df['intent'].unique()
        self.intent_to_id = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.id_to_intent = {idx: intent for intent, idx in self.intent_to_id.items()}
        
        # Convert intents to IDs
        df['label_id'] = df['intent'].map(self.intent_to_id)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['text'].tolist(),
            df['label_id'].tolist(),
            test_size=0.2,
            random_state=42,
            stratify=df['label_id']
        )
        
        return X_train, X_test, y_train, y_test
    
    def train_model(self, X_train, y_train, X_test, y_test):
        """Train mBERT model for intent classification"""
        # Initialize model
        num_labels = len(self.intent_to_id)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels
        )
        
        # Create datasets
        train_dataset = MedicalIntentDataset(X_train, y_train, self.tokenizer)
        test_dataset = MedicalIntentDataset(X_test, y_test, self.tokenizer)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
            acc = accuracy_score(labels, predictions)
            return {
                'accuracy': acc,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )
        
        # Train model
        trainer.train()
        
        # Save model
        model.save_pretrained('./mbert_medical_intent')
        self.tokenizer.save_pretrained('./mbert_medical_intent')
        
        # Save mappings
        with open('./mbert_medical_intent/intent_mappings.json', 'w') as f:
            json.dump({
                'intent_to_id': self.intent_to_id,
                'id_to_intent': self.id_to_intent
            }, f)
        
        return model, trainer
    
    def evaluate_model(self, model, X_test, y_test):
        """Evaluate trained model"""
        test_dataset = MedicalIntentDataset(X_test, y_test, self.tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=8)
        
        model.eval()
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = {
                    'input_ids': batch['input_ids'],
                    'attention_mask': batch['attention_mask']
                }
                
                outputs = model(**inputs)
                logits = outputs.logits
                
                predictions.extend(torch.argmax(logits, dim=-1).cpu().numpy())
                true_labels.extend(batch['labels'].cpu().numpy())
        
        # Generate classification report
        intent_names = [self.id_to_intent[i] for i in range(len(self.id_to_intent))]
        report = classification_report(true_labels, predictions, target_names=intent_names)
        
        return report, predictions, true_labels

def main():
    # Initialize trainer
    trainer = MBERTTrainer()
    
    # Prepare data (make sure medical_chatbot_dataset.json exists)
    try:
        X_train, X_test, y_train, y_test = trainer.prepare_data('medical_chatbot_dataset.json')
        print(f"Training data: {len(X_train)} samples")
        print(f"Test data: {len(X_test)} samples")
        
        # Train model
        print("Starting training...")
        model, trained_model = trainer.train_model(X_train, y_train, X_test, y_test)
        
        # Evaluate model
        print("Evaluating model...")
        report, predictions, true_labels = trainer.evaluate_model(model, X_test, y_test)
        print("\nClassification Report:")
        print(report)
        
    except FileNotFoundError:
        print("Dataset file not found. Please run dataset_builder.py first.")

if __name__ == "__main__":
    main()

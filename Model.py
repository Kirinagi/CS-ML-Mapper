import argilla as rg
from setfit import SetFitModel, SetFitTrainer
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
import re
import nltk
import gputil
import psutil
from nltk.corpus import stopwords

# Download NLTK resources
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
except:
    print("⚠ Warning: Could not download NLTK resources. Preprocessing may be limited.")

# ================================
# CONFIGURATION SECTION - UPDATE THESE PATHS AND COLUMN NAMES
# ================================

# DATA CONFIGURATION
DATA_FILE_PATH = "D:/Work Documents/denormalized_Data.XLSX"  # Your main data file
TEXT_COLUMN = "Description"  # Column containing the text to classify
LABEL_COLUMN = "Category"  # Column containing the labels/categories
OUTPUT_PREPROCESSED_FILE = "normalized_feedback.csv"  # Where to save preprocessed data

# TRAIN/TEST SPLIT CONFIGURATION
TEST_SIZE = 0.2  # 20% for testing, 80% for training
RANDOM_STATE = 42
STRATIFY = True  # Ensure balanced split across categories

# ARGILLA CONFIGURATION
ARGILLA_API_URL = "http://localhost:6900"
ARGILLA_API_KEY = "argilla.apikey"
ARGILLA_WORKSPACE = "argilla"
SKIP_ARGILLA_LABELING = False  # Set to False if you want to use Argilla for manual labeling

# MODEL CONFIGURATION
MODEL_NAME = "cahya/bert-base-indonesian-522M"
MODEL_SAVE_PATH = "./indonesian-complaint-classifier"

# SYSTEM CONFIGURATION
USE_GPU_OPTIMIZED_CONFIG = False  # Set to True for RTX 4070, False for Intel i3-1215U
AUTO_DETECT_SYSTEM = True  # Automatically detect system capabilities

# PREPROCESSING CONFIGURATION
USE_SLANG_NORMALIZATION = True
REMOVE_STOP_WORDS = False  # Set to True if you want to remove stop words
SLANG_DICT_URL = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

# ================================
# PREPROCESSING FUNCTIONS
# ================================

def load_slang_dictionary():
    """Load Indonesian slang dictionary from GitHub."""
    try:
        indo_slang_word = pd.read_csv(SLANG_DICT_URL)
        slang_dict = dict(zip(indo_slang_word['slang'], indo_slang_word['formal']))
        print(f"✓ Loaded {len(slang_dict)} slang words")
        return slang_dict
    except Exception as e:
        print(f"⚠ Warning: Could not load slang dictionary: {e}")
        return {}

def casefolding(text):
    """Converts text to lowercase and removes unnecessary characters."""
    if not isinstance(text, str):
        text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[-+]?[0-9]+', '', text)  # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.strip()
    return text

def normalize_slang_words(text, slang_dict):
    """Replaces slang words with their formal equivalents using the slang dictionary."""
    if not slang_dict:
        return text
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)

def remove_stop_words(text):
    """Removes stop words from the text."""
    try:
        stop_words = set(stopwords.words('indonesian'))
        words = nltk.word_tokenize(text)
        return ' '.join([word for word in words if word not in stop_words])
    except:
        print("⚠ Warning: Could not remove stop words. Returning original text.")
        return text

def text_preprocessing_process(text, slang_dict=None):
    """Main text preprocessing pipeline."""
    text = casefolding(text)
    
    if USE_SLANG_NORMALIZATION and slang_dict:
        text = normalize_slang_words(text, slang_dict)
    
    if REMOVE_STOP_WORDS:
        text = remove_stop_words(text)
    
    return text

# ================================
# DATA LOADING AND PREPROCESSING
# ================================

def load_and_preprocess_data():
    """Load data from file and apply preprocessing."""
    print("📊 Loading and preprocessing data...")
    
    # Load slang dictionary
    slang_dict = load_slang_dictionary() if USE_SLANG_NORMALIZATION else {}
    
    # Load main data file
    try:
        if DATA_FILE_PATH.endswith('.xlsx') or DATA_FILE_PATH.endswith('.xls'):
            df = pd.read_excel(DATA_FILE_PATH, engine='openpyxl')
        elif DATA_FILE_PATH.endswith('.csv'):
            df = pd.read_csv(DATA_FILE_PATH)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
        
        print(f"✓ Loaded {len(df)} records from {DATA_FILE_PATH}")
        
    except FileNotFoundError:
        print(f"✗ Error: The file '{DATA_FILE_PATH}' was not found.")
        return None
    except Exception as e:
        print(f"✗ An error occurred while loading data: {e}")
        return None
    
    # Check if required columns exist
    if TEXT_COLUMN not in df.columns:
        print(f"✗ Text column '{TEXT_COLUMN}' not found in file")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    if LABEL_COLUMN not in df.columns:
        print(f"✗ Label column '{LABEL_COLUMN}' not found in file")
        print(f"Available columns: {df.columns.tolist()}")
        return None
    
    # Clean data
    df = df.dropna(subset=[TEXT_COLUMN, LABEL_COLUMN])
    df = df[df[TEXT_COLUMN].str.strip() != '']
    df = df[df[LABEL_COLUMN].str.strip() != '']
    
    print(f"✓ After cleaning: {len(df)} records")
    
    # Apply preprocessing
    print("🔄 Applying text preprocessing...")
    df['normalized_text'] = df[TEXT_COLUMN].apply(
        lambda x: text_preprocessing_process(x, slang_dict)
    )
    
    # Remove empty preprocessed texts
    df = df[df['normalized_text'].str.strip() != '']
    
    print(f"✓ After preprocessing: {len(df)} records")
    
    # Save preprocessed data
    try:
        df.to_csv(OUTPUT_PREPROCESSED_FILE, index=False)
        print(f"✓ Preprocessed data saved to: {OUTPUT_PREPROCESSED_FILE}")
    except Exception as e:
        print(f"⚠ Warning: Could not save preprocessed data: {e}")
    
    # Show sample of preprocessing results
    print("\n📋 Sample preprocessing results:")
    for i in range(min(3, len(df))):
        print(f"Original: {df.iloc[i][TEXT_COLUMN]}")
        print(f"Preprocessed: {df.iloc[i]['normalized_text']}")
        print(f"Label: {df.iloc[i][LABEL_COLUMN]}")
        print("-" * 50)
    
    return df

def split_data(df):
    """Split data into train and test sets."""
    print("\n✂️ Splitting data into train and test sets...")
    
    texts = df['normalized_text'].tolist()
    labels = df[LABEL_COLUMN].tolist()
    
    # Get label distribution
    label_counts = pd.Series(labels).value_counts()
    print("📊 Label distribution:")
    for label, count in label_counts.items():
        print(f"  {label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Split data
    stratify_labels = labels if STRATIFY else None
    
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=stratify_labels
        )
        
        print(f"✓ Training set: {len(X_train)} samples")
        print(f"✓ Test set: {len(X_test)} samples")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        print(f"✗ Error splitting data: {e}")
        return None, None, None, None

# ================================
# ARGILLA INTEGRATION (OPTIONAL)
# ================================

def setup_argilla_dataset(X_train, y_train):
    """Set up Argilla dataset for manual labeling (optional)."""
    if SKIP_ARGILLA_LABELING:
        print("⏭️ Skipping Argilla labeling (SKIP_ARGILLA_LABELING=True)")
        return X_train, y_train
    
    print("\n📝 Setting up Argilla dataset...")
    
    try:
        # Initialize Argilla
        rg.init(
            api_url=ARGILLA_API_URL,
            api_key=ARGILLA_API_KEY,
            workspace=ARGILLA_WORKSPACE
        )
        
        # Get unique labels
        unique_labels = sorted(list(set(y_train)))
        
        # Create dataset settings
        settings = rg.Settings(
            fields=[
                rg.TextField(name="text", title="Complaint Text", use_markdown=False),
                rg.TextField(name="source", title="Source", required=False),
            ],
            questions=[
                rg.LabelQuestion(
                    name="cs_ticket_type",
                    title="Ticket Type",
                    labels=unique_labels,
                    required=True
                )
            ],
            guidelines="Please classify Indonesian complaints into the appropriate category based on the main issue described.",
            metadata=[
                rg.TermsMetadataProperty(name="language", values=["indonesian"]),
                rg.IntegerMetadataProperty(name="length"),
            ]
        )
        
        # Create or get dataset
        try:
            dataset = rg.Dataset(name="indonesian-complaint-classification", settings=settings)
            dataset.create()
            print("✓ Created new Argilla dataset")
        except:
            dataset = rg.Dataset(name="indonesian-complaint-classification")
            print("✓ Using existing Argilla dataset")
        
        # Create records
        records = []
        for text, label in zip(X_train, y_train):
            record = rg.Record(
                fields={
                    "text": text,
                    "source": "training_data"
                },
                suggestions=[
                    rg.Suggestion(
                        question_name="cs_ticket_type",
                        value=label,
                        score=0.9
                    )
                ],
                metadata={
                    "language": "indonesian",
                    "length": len(text)
                }
            )
            records.append(record)
        
        # Log records
        dataset.records.log(records)
        print(f"✓ Added {len(records)} records to Argilla dataset")
        
        # Wait for manual labeling
        print(f"\n🏷️ Please go to {ARGILLA_API_URL} to review and label your data")
        print("Press Enter when you've finished labeling...")
        input()
        
        # Export labeled data
        labeled_dataset = dataset.to_datasets()
        
        if labeled_dataset and len(labeled_dataset) > 0:
            # Convert back to lists
            texts = []
            labels = []
            for record in labeled_dataset:
                if 'cs_ticket_type' in record and record['cs_ticket_type'] is not None:
                    texts.append(record['text'])
                    labels.append(record['cs_ticket_type'])
            
            print(f"✓ Using {len(texts)} manually labeled examples")
            return texts, labels
        else:
            print("⚠ No manually labeled data found. Using original split.")
            return X_train, y_train
            
    except Exception as e:
        print(f"✗ Error with Argilla setup: {e}")
        print("⚠ Proceeding with original training data")
        return X_train, y_train

# ================================
# MODEL TRAINING
# ================================

def train_model(X_train, y_train, X_test, y_test):
    """Train the SetFit model."""
    print("\n🤖 Training SetFit model...")
    
    # Create label mapping
    unique_labels = sorted(list(set(y_train + y_test)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    id_to_label = {i: label for label, i in label_to_id.items()}
    
    print(f"✓ Found {len(unique_labels)} unique labels: {unique_labels}")
    
    # Convert labels to integers
    y_train_ids = [label_to_id[label] for label in y_train]
    y_test_ids = [label_to_id[label] for label in y_test]
    
    # Create datasets
    train_dataset = Dataset.from_dict({
        'text': X_train,
        'label': y_train_ids
    })
    
    test_dataset = Dataset.from_dict({
        'text': X_test,
        'label': y_test_ids
    })
    
    # Initialize model
    try:
        model = SetFitModel.from_pretrained(
            MODEL_NAME,
            use_differentiable_head=True,
            head_params={"out_features": len(unique_labels)}
        )
        print("✓ Model initialized successfully")
    except Exception as e:
        print(f"✗ Error initializing model: {e}")
        return None, None, None
    
    # Create trainer
    # trainer = SetFitTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     eval_dataset=test_dataset,
    #     loss_class="CosineSimilarityLoss",
    #     metric="accuracy",
    #     batch_size=16,
    #     num_iterations=20,
    #     num_epochs=1,
    #     column_mapping={"text": "text", "label": "label"}
    # )

    trainer = SetFitTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    
    # === CORE TRAINING PARAMETERS ===
    num_iterations=40,        # 20-60 optimal for 30k samples
    num_epochs=3,             # 1-5 for classification head
    
    # === BATCH CONFIGURATION ===
    batch_size=32,            # 16-64 depending on GPU memory
    
    # === LEARNING RATES ===
    learning_rate=2e-5,       # Conservative for Indonesian BERT
    body_learning_rate=2e-5,  # Same as above (sentence transformer)
    head_learning_rate=2e-3,  # Higher for classification head
    
    # === LOSS AND OPTIMIZATION ===
    loss_class="CosineSimilarityLoss",  # Best for SetFit
    optimizer="AdamW",
    warmup_proportion=0.1,    # 10% warmup steps
    
    # === SAMPLING STRATEGY ===
    sampling_strategy="oversampling",  # Handle class imbalance
    
    # === EVALUATION ===
    metric="accuracy",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    
    # === COLUMN MAPPING ===
    column_mapping={"text": "text", "label": "label"}
)
    
    # Train model
    try:
        trainer.train()
        print("✓ Training completed successfully")
    except Exception as e:
        print(f"✗ Error during training: {e}")
        return None, None, None
    
    # Evaluate model
    print("\n📊 Evaluating model...")
    predictions = model.predict(X_test)
    
    print("\nClassification Report:")
    print(classification_report(y_test_ids, predictions, target_names=unique_labels))
    
    # Save model
    try:
        model.save_pretrained(MODEL_SAVE_PATH)
        print(f"✓ Model saved to: {MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")
    
    return model, unique_labels, id_to_label

def test_model(model, unique_labels, id_to_label):
    """Test the model with sample predictions."""
    print("\n🧪 Testing model with sample data...")
    
    # Sample test cases
    test_samples = [
        "Operator telepon tidak membantu sama sekali dan sangat lambat",
        "Bagaimana cara mengganti password akun saya?",
        "Bisakah saya mendapatkan informasi tentang promo terbaru?",
        "Produk yang diterima rusak dan tidak sesuai dengan gambar",
        "Aplikasi sering crash dan tidak bisa login"
    ]
    
    # Apply same preprocessing
    slang_dict = load_slang_dictionary() if USE_SLANG_NORMALIZATION else {}
    preprocessed_samples = [
        text_preprocessing_process(text, slang_dict) for text in test_samples
    ]
    
    # Make predictions
    predictions = model.predict(preprocessed_samples)
    predicted_labels = [id_to_label[pred] for pred in predictions]
    
    print("\n📋 Sample Predictions:")
    for i, (original, preprocessed, pred) in enumerate(zip(test_samples, preprocessed_samples, predicted_labels)):
        print(f"Test {i+1}:")
        print(f"  Original: {original}")
        print(f"  Preprocessed: {preprocessed}")
        print(f"  Predicted: {pred}")
        print("-" * 50)

# ================================
# MAIN EXECUTION
# ================================

def main():
    print("🚀 Starting Integrated Indonesian Complaint Classification Pipeline")
    print("=" * 70)
    
    # Step 1: Load and preprocess data
    df = load_and_preprocess_data()
    if df is None:
        print("❌ Cannot proceed without data")
        return
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(df)
    if X_train is None:
        print("❌ Cannot proceed without proper data split")
        return
    
    # Step 3: Optional Argilla labeling
    X_train, y_train = setup_argilla_dataset(X_train, y_train)
    
    # Step 4: Train model
    model, unique_labels, id_to_label = train_model(X_train, y_train, X_test, y_test)
    if model is None:
        print("❌ Training failed")
        return
    
    # Step 5: Test model
    test_model(model, unique_labels, id_to_label)
    
    print("\n🎉 Pipeline completed successfully!")
    print(f"📁 Model saved to: {MODEL_SAVE_PATH}")
    print(f"📁 Preprocessed data saved to: {OUTPUT_PREPROCESSED_FILE}")

if __name__ == "__main__":
    main()
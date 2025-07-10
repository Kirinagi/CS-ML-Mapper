import pandas as pd
from setfit import SetFitModel
import re
import json
import os
from tqdm import tqdm # A library to show a progress bar

# ================================
# CONFIGURATION
# ================================
# --- File Paths ---
INPUT_FILE_PATH = "new_data_to_predict.csv" # The new data you want to predict on
OUTPUT_FILE_PATH = "prediction_output.csv"  # Where to save the results

# --- Column Name ---
DESCRIPTION_COLUMN = "DESCRIPTION" # The column in your input file with the text to classify

# --- Model & Mapping Paths (must match training script) ---
MODEL_PATHS = {
    "type": "./indonesian-classifier-type",
    "explanation": "./indonesian-classifier-explanation"
}
MAPPING_PATHS = {
    "explanation_to_work_order": "./explanation_to_work_order.json",
    "work_order_to_category": "./work_order_to_category.json"
}
SLANG_DICT_URL = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"


# ================================
# SETUP: LOAD MODELS & MAPPINGS
# ================================
def load_assets():
    """Load all trained models, mappings, and slang dictionary from disk."""
    print("🧠 Loading models, mappings, and dictionaries...")
    
    # Load Models
    models = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.isdir(path):
            print(f"✗ FATAL: Model directory not found at {path}. Please run train_pipeline.py first.")
            return None, None
        models[name] = SetFitModel.from_pretrained(path)
        print(f"✓ Loaded model '{name}'")

    # Load Mappings
    mappings = {}
    for name, path in MAPPING_PATHS.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                mappings[name] = json.load(f)
            print(f"✓ Loaded mapping '{name}'")
        except FileNotFoundError:
            print(f"✗ FATAL: Mapping file not found at {path}. Please run train_pipeline.py first.")
            return None, None
            
    # Load Slang Dictionary
    try:
        df_slang = pd.read_csv(SLANG_DICT_URL)
        slang_dict = dict(zip(df_slang['slang'], df_slang['formal']))
        print(f"✓ Loaded {len(slang_dict)} slang words")
    except Exception as e:
        print(f"⚠ Warning: Could not load slang dictionary: {e}")
        slang_dict = {}

    return models, mappings, slang_dict

# Preprocessing function (must be identical to the one used in training)
def text_preprocessing_process(text, slang_dict):
    if not isinstance(text, str): text = str(text)
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return ' '.join(normalized_words)


# ================================
# BATCH PREDICTION
# ================================
def run_batch_predictions(df, models, mappings, slang_dict):
    """
    Processes a DataFrame to add prediction columns.
    """
    # Preprocess all descriptions at once for efficiency
    print("\n🔄 Preprocessing all descriptions...")
    preprocessed_texts = [text_preprocessing_process(text, slang_dict) for text in df[DESCRIPTION_COLUMN]]

    print("🤖 Making predictions with the models...")
    # Get predictions from the ML models in batches
    predicted_types = models["type"].predict(preprocessed_texts)
    predicted_explanations = models["explanation"].predict(preprocessed_texts)

    # Use mappings to get the final categories
    print("🗺️ Applying mappings...")
    predicted_work_orders = [mappings["explanation_to_work_order"].get(exp, "Unknown") for exp in predicted_explanations]
    predicted_categories = [mappings["work_order_to_category"].get(wo, "Unknown") for wo in predicted_work_orders]

    # Add the new predictions as columns to the DataFrame
    df['predicted_Type'] = predicted_types
    df['predicted_Explaination'] = predicted_explanations
    df['predicted_Work_Order_Category'] = predicted_work_orders
    df['predicted_Category'] = predicted_categories
    
    return df

# ================================
# MAIN EXECUTION
# ================================
if __name__ == "__main__":
    # 1. Load the assets
    models, mappings, slang_dict = load_assets()

    if models and mappings:
        # 2. Load the input data file
        try:
            print(f"\n📂 Loading data from '{INPUT_FILE_PATH}'...")
            # Use a progress bar for large files by installing `tqdm`: pip install tqdm
            tqdm.pandas()
            input_df = pd.read_csv(INPUT_FILE_PATH)
            print(f"✓ Found {len(input_df)} rows to process.")
        except FileNotFoundError:
            print(f"✗ ERROR: Input file not found at '{INPUT_FILE_PATH}'. Please check the path and file name.")
            exit()
        except Exception as e:
            print(f"✗ ERROR: Could not read the input file. Error: {e}")
            exit()

        # 3. Run the prediction pipeline on the DataFrame
        result_df = run_batch_predictions(input_df, models, mappings, slang_dict)

        # 4. Save the results to a new CSV
        try:
            result_df.to_csv(OUTPUT_FILE_PATH, index=False, encoding='utf-8-sig')
            print(f"\n🎉 Success! Predictions saved to '{OUTPUT_FILE_PATH}'")
        except Exception as e:
            print(f"✗ ERROR: Could not save the output file. Error: {e}")
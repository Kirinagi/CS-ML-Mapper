import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import json
import os
from tqdm import tqdm
from dotenv import load_dotenv
from odps import ODPS
from odps.df import DataFrame as ODPSDataFrame
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ================================
# PANDAS DISPLAY CONFIGURATION
# ================================

# Configure pandas to display large integers properly
pd.set_option('display.float_format', '{:.0f}'.format)
pd.set_option('display.precision', 0)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# ================================
# CONFIGURATION
# ================================

load_dotenv()

# --- MaxCompute/ODPS Configuration ---
ODPS_CONFIG = {
    "access_id": os.getenv("ODPS_ACCESS_ID"),
    "secret_access_key": os.getenv("ODPS_SECRET_ACCESS_KEY"),
    "project": os.getenv("ODPS_PROJECT"),
    "endpoint": os.getenv("ODPS_ENDPOINT")
}

# --- Source Table/Query Configuration ---
SOURCE_QUERY = """
SELECT
    CAST(id AS STRING) as id, 
    CAST(customer_id AS STRING) as customer_id, 
    CAST(phone_number AS STRING) as phone_number, 
    CAST(have_phone_number AS STRING) as have_phone_number, 
    CAST(app_version AS STRING) as app_version,
    feedback_type, description, mobile_model, whatsapp, submit_time,
    fixed_or_not, notes, created_time, created_by, updated_time,
    updated_by, delete_time, delete_by, is_delete, version,
    remark, ext, history_flag, migration_time, transfer_work_order,
    customer_email, customer_name
FROM kredi_dwd.dwd_customer_support_customer_feedback_df
WHERE pt = MAX_PT('kredi_dwd.dwd_customer_support_customer_feedback_df')
    AND to_date(submit_time) = DATEADD(to_date(GETDATE()), -1, 'dd')
    AND fixed_or_not = 0
ORDER BY submit_time DESC
"""

# --- Output Table Configuration ---
OUTPUT_TABLE_NAME = "dwd_customer_support_customer_feedback_hftransformer_df"
OUTPUT_PARTITION_TEMPLATE = "pt='{date}'" 

# --- Column Name ---
DESCRIPTION_COLUMN = "description"

# --- Model & Mapping Paths ---
MODEL_PATHS = {
    # "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
    # "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
    "type": "./models/model_type_ft",
    "explanation": "./models/model_explanation_ft"
}

MAPPING_PATHS = {
    "type": os.path.join(MODEL_PATHS["type"], 'config.json'),
    "explanation": os.path.join(MODEL_PATHS["explanation"], 'config.json'),
    "explanation_to_work_order": ".\\explanation_to_work_order.json",
    "work_order_to_category": ".\\work_order_to_category.json",
    "work_order_to_work_priority": ".\\work_order_to_work_priority.json",
}

# ================================
# MAXCOMPUTE/ODPS FUNCTIONS
# ================================

def create_odps_connection():
    """Creates and returns a MaxCompute ODPS connection object."""
    try:
        odps = ODPS(
            access_id=ODPS_CONFIG["access_id"],
            secret_access_key=ODPS_CONFIG["secret_access_key"],
            project=ODPS_CONFIG["project"],
            endpoint=ODPS_CONFIG["endpoint"]
        )
        print(f"✓ Connected to MaxCompute project: {ODPS_CONFIG['project']}")
        return odps
    except Exception as e:
        print(f"✗ Error connecting to MaxCompute: {e}")
        return None

def fetch_data_from_odps(odps, query=None):
    """Fetches data from MaxCompute by executing a query."""
    if not odps:
        return None
    try:
        print("📊 Executing ODPS SQL query...")
        with odps.execute_sql(query).open_reader() as reader:
            df = reader.to_pandas()
            
            # ================================
            # FIX CUSTOMER_ID DISPLAY ISSUE
            # ================================
            
            # Convert customer_id and other large integer columns to string to preserve exact values
            large_int_columns = ['id', 'customer_id']
            for col in large_int_columns:
                if col in df.columns:
                    # Convert to string to preserve exact integer value
                    df[col] = df[col].astype(str)
                    # Remove any '.0' suffix if present
                    df[col] = df[col].str.replace(r'\.0$', '', regex=True)
                    print(f"✓ Fixed display format for column: {col}")
            
            return df
    except Exception as e:
        print(f"✗ Error fetching data from MaxCompute: {e}")
        return None

def create_target_table_if_not_exists(odps_conn, table_name):
    """
    Creates the target table if it doesn't exist.
    
    Args:
        odps_conn: The ODPS connection object.
        table_name: The name of the target table to create.
    """
    try:
        # Check if table exists
        if odps_conn.exist_table(table_name):
            print(f"✓ Table '{table_name}' already exists.")
            return True
        
        print(f"📋 Creating table '{table_name}'...")
        
        # Define the table schema - using STRING for large IDs to avoid precision loss
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id STRING,
            customer_id STRING,
            phone_number STRING,
            have_phone_number BIGINT,
            app_version STRING,
            feedback_type STRING,
            description STRING,
            mobile_model STRING,
            whatsapp BIGINT,
            submit_time DATETIME,
            fixed_or_not BIGINT,
            notes STRING,
            created_time DATETIME,
            created_by STRING,
            updated_time DATETIME,
            updated_by STRING,
            delete_time DATETIME,
            delete_by STRING,
            is_delete BIGINT,
            version BIGINT,
            remark STRING,
            ext STRING,
            history_flag BIGINT,
            migration_time DATETIME,
            transfer_work_order STRING,
            customer_email STRING,
            customer_name STRING,
            predicted_type STRING,
            predicted_explanation STRING,
            work_order STRING,
            category STRING,
            work_priority STRING
        )
        PARTITIONED BY (pt STRING)
        STORED AS ALIORC
        TBLPROPERTIES ('transactional'='true')
        """
        
        # Execute the CREATE TABLE statement
        odps_conn.execute_sql(create_table_sql)
        print(f"✓ Table '{table_name}' created successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Error creating table '{table_name}': {e}")
        return False

def upload_to_odps(odps_conn, df, table_name, partition_spec, overwrite=True):
    """
    Uploads a pandas DataFrame to a partitioned MaxCompute table.
    Creates the table if it doesn't exist.
    
    Args:
        odps_conn: The ODPS connection object.
        df: The pandas DataFrame to upload.
        table_name: The name of the target ODPS table.
        partition_spec: The partition specification string (e.g., "pt='20250801'").
        overwrite: If True, overwrite the partition if it exists.
    """
    if df.empty:
        print("DataFrame is empty. Nothing to upload.")
        return

    try:
        # First, ensure the table exists
        if not create_target_table_if_not_exists(odps_conn, table_name):
            print(f"✗ Failed to create or verify table '{table_name}'. Upload aborted.")
            return
        
        print(f"💾 Writing {len(df)} rows to ODPS table '{table_name}' with partition: {partition_spec}")
        
        # Convert DataFrame columns to match expected types
        df_upload = df.copy()
        
        # Handle datetime columns - ensure they're in the right format and remove timezone info
        datetime_columns = ['submit_time', 'created_time', 'updated_time', 'delete_time', 'migration_time']
        for col in datetime_columns:
            if col in df_upload.columns:
                # Convert to datetime and remove timezone information to avoid PyArrow issues
                df_upload[col] = pd.to_datetime(df_upload[col], errors='coerce')
                if hasattr(df_upload[col].dtype, 'tz') and df_upload[col].dt.tz is not None:
                    # Convert timezone-aware datetime to naive datetime (UTC)
                    df_upload[col] = df_upload[col].dt.tz_convert('UTC').dt.tz_localize(None)
                # Ensure datetime format is compatible with MaxCompute
                df_upload[col] = df_upload[col].dt.strftime('%Y-%m-%d %H:%M:%S')
                df_upload[col] = pd.to_datetime(df_upload[col], errors='coerce')
        
        # Handle integer columns
        int_columns = ['have_phone_number', 'whatsapp', 'fixed_or_not', 'is_delete', 'version', 'history_flag']
        for col in int_columns:
            if col in df_upload.columns:
                df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0).astype('Int64')
        
        # Ensure all string columns are properly formatted
        string_columns = ['id', 'customer_id', 'phone_number', 'app_version', 'feedback_type', 
                         'description', 'mobile_model', 'notes', 'created_by', 'updated_by', 
                         'delete_by', 'remark', 'ext', 'transfer_work_order', 'customer_email', 
                         'customer_name', 'predicted_type', 'predicted_explanation', 'work_order', 
                         'category', 'work_priority']
        for col in string_columns:
            if col in df_upload.columns:
                # For ID columns, ensure they remain as strings without scientific notation
                if col in ['id', 'customer_id']:
                    df_upload[col] = df_upload[col].astype(str)
                    # Remove any '.0' suffix that might have been added
                    df_upload[col] = df_upload[col].str.replace(r'\.0$', '', regex=True)
                else:
                    df_upload[col] = df_upload[col].astype(str).fillna('')
        
        # The write_table method handles creating the partition if it doesn't exist
        # and writing the DataFrame data directly.
        odps_conn.write_table(
            table_name, 
            df_upload, 
            partition=partition_spec, 
            overwrite=overwrite, 
            create_partition=True
        )
        print("✓ Upload successful!")
        
    except Exception as e:
        print(f"✗ Error uploading data to MaxCompute: {e}")
        # Print more detailed error information
        import traceback
        print(f"Detailed error: {traceback.format_exc()}")

# ================================
# HELPER FUNCTIONS
# ================================

def load_model_and_tokenizer(model_path):
    """Loads a Hugging Face model and tokenizer from a specified path."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        print(f"✓ Model loaded from {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"✗ Error loading model from {model_path}: {e}")
        return None, None

def load_json_mapping(file_path):
    """Loads a JSON mapping file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"✗ Error loading JSON from {file_path}: {e}")
        return {}

def predict_batch(model, tokenizer, texts, batch_size=32):
    """Performs batch prediction on a list of texts."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="🤖 Predicting"):
            batch_texts = texts[i:i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
    return all_preds

def map_predictions_to_labels(predictions, config_path):
    """Maps prediction IDs to their corresponding string labels."""
    try:
        with open(config_path) as f:
            config = json.load(f)
        id2label = config['id2label']
        return [id2label.get(str(p), "UNKNOWN") for p in predictions]
    except Exception as e:
        print(f"✗ Error mapping predictions: {e}")
        return ["ERROR"] * len(predictions)

# ================================
# MAIN EXECUTION
# ================================

def main():
    """Main function to execute the entire pipeline."""
    print("🚀 Starting the feedback processing pipeline...")

    # --- 1. Fetch Data ---
    odps_connection = create_odps_connection()
    if not odps_connection:
        return
    df = fetch_data_from_odps(odps_connection, query=SOURCE_QUERY)
    if df is None or df.empty:
        print("✗ No data fetched or dataframe is empty. Exiting.")
        return
    print(f"✓ Fetched {len(df)} records from ODPS.")
    
    # Display sample of customer_id to verify it's displaying correctly
    print(f"\n📋 Sample customer_id values:")
    if 'customer_id' in df.columns:
        sample_ids = df['customer_id'].head(5).tolist()
        for i, cid in enumerate(sample_ids, 1):
            print(f"  {i}. {cid} (type: {type(cid)})")

    # --- 2. Load Models & Mappings ---
    print("\n🧠 Loading models, tokenizers, and mappings...")
    model_type, tokenizer_type = load_model_and_tokenizer(MODEL_PATHS["type"])
    model_explanation, tokenizer_explanation = load_model_and_tokenizer(MODEL_PATHS["explanation"])
    explanation_to_work_order = load_json_mapping(MAPPING_PATHS["explanation_to_work_order"])
    work_order_to_category = load_json_mapping(MAPPING_PATHS["work_order_to_category"])
    work_order_to_priority = load_json_mapping(MAPPING_PATHS["work_order_to_work_priority"])

    if not all([model_type, model_explanation, explanation_to_work_order, work_order_to_category, work_order_to_priority]):
        print("✗ Failed to load one or more models or mappings. Exiting.")
        return

    # --- 3. Preprocess and Predict ---
    descriptions = df[DESCRIPTION_COLUMN].tolist()

    # Predict 'Type'
    predicted_type_ids = predict_batch(model_type, tokenizer_type, descriptions)
    df['predicted_type'] = map_predictions_to_labels(predicted_type_ids, MAPPING_PATHS["type"])

    # Predict 'Explanation'
    predicted_explanation_ids = predict_batch(model_explanation, tokenizer_explanation, descriptions)
    df['predicted_explanation'] = map_predictions_to_labels(predicted_explanation_ids, MAPPING_PATHS["explanation"])

    # --- 4. Map Predictions to Final Categories ---
    print("\n🔄 Mapping predictions to final work order categories...")
    df['work_order'] = df['predicted_explanation'].map(explanation_to_work_order.get)
    df['category'] = df['work_order'].map(work_order_to_category.get)
    df['work_priority'] = df['work_order'].map(work_order_to_priority.get)

    # --- 5. Prepare and Upload Results ---
    # Define the columns you want to upload to the new table.
    columns_to_upload = [
        # Original columns from the source query
        'id', 
        'customer_id', 
        'phone_number', 
        'have_phone_number', 
        'app_version',
        'feedback_type', 
        'description', 
        'mobile_model', 
        'whatsapp', 
        'submit_time',
        'fixed_or_not', 
        'notes', 
        'created_time', 
        'created_by', 
        'updated_time',
        'updated_by', 
        'delete_time', 
        'delete_by', 
        'is_delete', 
        'version',
        'remark', 
        'ext', 
        'history_flag', 
        'migration_time', 
        'transfer_work_order',
        'customer_email', 
        'customer_name',
        # New predicted columns added by the ML pipeline
        'predicted_type', 
        'predicted_explanation', 
        'work_order', 
        'category', 
        'work_priority'
    ]
    
    # Ensure all selected columns exist in the DataFrame before uploading
    df_to_upload = df[[col for col in columns_to_upload if col in df.columns]].copy()

    # Generate the dynamic partition string using today's date
    partition_date = datetime.now().strftime('%Y%m%d')
    dynamic_partition = OUTPUT_PARTITION_TEMPLATE.format(date=partition_date)
    
    print("\n✨ Final Processed Data (Top 5 rows):")
    # Display with proper formatting for large integers
    display_df = df_to_upload.head()
    print(display_df.to_string(index=False, max_colwidth=50))

    # Upload the processed DataFrame to MaxCompute
    upload_to_odps(odps_connection, df_to_upload, OUTPUT_TABLE_NAME, dynamic_partition)
    
    print("\n✅ Pipeline finished successfully!")

if __name__ == '__main__':
    main()
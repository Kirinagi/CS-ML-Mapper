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
    "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
    "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
}

MAPPING_PATHS = {
    "type": os.path.join(MODEL_PATHS["type"], 'config.json'),
    "explanation": os.path.join(MODEL_PATHS["explanation"], 'config.json'),
    "explanation_to_work_order": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\explanation_to_work_order.json",
    "work_order_to_category": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_category.json",
    "work_order_to_work_priority": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_work_priority.json",
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



# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import json
# import os
# from tqdm import tqdm
# from dotenv import load_dotenv
# from odps import ODPS
# from odps.df import DataFrame as ODPSDataFrame
# from datetime import datetime
# import warnings

# warnings.filterwarnings('ignore')

# # ================================
# # CONFIGURATION
# # ================================

# load_dotenv()

# # --- MaxCompute/ODPS Configuration ---
# ODPS_CONFIG = {
#     "access_id": os.getenv("ODPS_ACCESS_ID"),
#     "secret_access_key": os.getenv("ODPS_SECRET_ACCESS_KEY"),
#     "project": os.getenv("ODPS_PROJECT"),
#     "endpoint": os.getenv("ODPS_ENDPOINT")
# }

# # --- Source Table/Query Configuration ---
# SOURCE_QUERY = """
# SELECT
#     id, customer_id, phone_number, have_phone_number, app_version,
#     feedback_type, description, mobile_model, whatsapp, submit_time,
#     fixed_or_not, notes, created_time, created_by, updated_time,
#     updated_by, delete_time, delete_by, is_delete, version,
#     remark, ext, history_flag, migration_time, transfer_work_order,
#     customer_email, customer_name
# FROM kredi_dwd.dwd_customer_support_customer_feedback_df
# WHERE pt = MAX_PT('kredi_dwd.dwd_customer_support_customer_feedback_df')
#     AND to_date(submit_time) = DATEADD(to_date(GETDATE()), -1, 'dd')
#     AND fixed_or_not = 0
# ORDER BY submit_time DESC
# """

# # --- Output Table Configuration ---
# # The target table should exist or be created with a schema that matches the DataFrame.
# OUTPUT_TABLE_NAME = "dwd_customer_support_customer_feedback_hftransformer_df"
# # The '${bizdate}' is a DataWorks parameter. We'll generate this dynamically.
# OUTPUT_PARTITION_TEMPLATE = "pt='{date}'" 

# # --- Column Name ---
# DESCRIPTION_COLUMN = "description"

# # --- Model & Mapping Paths ---
# MODEL_PATHS = {
#     "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
#     "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
# }

# MAPPING_PATHS = {
#     "type": os.path.join(MODEL_PATHS["type"], 'config.json'),
#     "explanation": os.path.join(MODEL_PATHS["explanation"], 'config.json'),
#     "explanation_to_work_order": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\explanation_to_work_order.json",
#     "work_order_to_category": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_category.json",
#     "work_order_to_work_priority": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_work_priority.json",
# }

# # ================================
# # MAXCOMPUTE/ODPS FUNCTIONS
# # ================================

# def create_odps_connection():
#     """Creates and returns a MaxCompute ODPS connection object."""
#     try:
#         odps = ODPS(
#             access_id=ODPS_CONFIG["access_id"],
#             secret_access_key=ODPS_CONFIG["secret_access_key"],
#             project=ODPS_CONFIG["project"],
#             endpoint=ODPS_CONFIG["endpoint"]
#         )
#         print(f"✓ Connected to MaxCompute project: {ODPS_CONFIG['project']}")
#         return odps
#     except Exception as e:
#         print(f"✗ Error connecting to MaxCompute: {e}")
#         return None

# def fetch_data_from_odps(odps, query=None):
#     """Fetches data from MaxCompute by executing a query."""
#     if not odps:
#         return None
#     try:
#         print("📊 Executing ODPS SQL query...")
#         with odps.execute_sql(query).open_reader() as reader:
#             df = reader.to_pandas()
#             return df
#     except Exception as e:
#         print(f"✗ Error fetching data from MaxCompute: {e}")
#         return None

# def create_target_table_if_not_exists(odps_conn, table_name):
#     """
#     Creates the target table if it doesn't exist.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         table_name: The name of the target table to create.
#     """
#     try:
#         # Check if table exists
#         if odps_conn.exist_table(table_name):
#             print(f"✓ Table '{table_name}' already exists.")
#             return True
        
#         print(f"📋 Creating table '{table_name}'...")
        
#         # Define the table schema - adjust data types as needed
#         create_table_sql = f"""
#         CREATE TABLE IF NOT EXISTS {table_name} (
#             id STRING,
#             customer_id STRING,
#             phone_number STRING,
#             have_phone_number BIGINT,
#             app_version STRING,
#             feedback_type STRING,
#             description STRING,
#             mobile_model STRING,
#             whatsapp BIGINT,
#             submit_time DATETIME,
#             fixed_or_not BIGINT,
#             notes STRING,
#             created_time DATETIME,
#             created_by STRING,
#             updated_time DATETIME,
#             updated_by STRING,
#             delete_time DATETIME,
#             delete_by STRING,
#             is_delete BIGINT,
#             version BIGINT,
#             remark STRING,
#             ext STRING,
#             history_flag BIGINT,
#             migration_time DATETIME,
#             transfer_work_order STRING,
#             customer_email STRING,
#             customer_name STRING,
#             predicted_type STRING,
#             predicted_explanation STRING,
#             work_order STRING,
#             category STRING,
#             work_priority STRING
#         )
#         PARTITIONED BY (pt STRING)
#         STORED AS ALIORC
#         TBLPROPERTIES ('transactional'='true')
#         """
        
#         # Execute the CREATE TABLE statement
#         odps_conn.execute_sql(create_table_sql)
#         print(f"✓ Table '{table_name}' created successfully!")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error creating table '{table_name}': {e}")
#         return False

# def create_table_from_dataframe(odps_conn, df, table_name):
#     """
#     Alternative method: Create table schema based on DataFrame dtypes.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         df: The pandas DataFrame to analyze for schema.
#         table_name: The name of the target table to create.
#     """
#     try:
#         if odps_conn.exist_table(table_name):
#             print(f"✓ Table '{table_name}' already exists.")
#             return True
        
#         print(f"📋 Creating table '{table_name}' from DataFrame schema...")
        
#         # Map pandas dtypes to MaxCompute types
#         dtype_mapping = {
#             'object': 'STRING',
#             'int64': 'BIGINT',
#             'int32': 'BIGINT',
#             'float64': 'DOUBLE',
#             'float32': 'DOUBLE',
#             'bool': 'BOOLEAN',
#             'datetime64[ns]': 'DATETIME',
#             'Int64': 'BIGINT'  # pandas nullable integer
#         }
        
#         columns_def = []
#         for col, dtype in df.dtypes.items():
#             odps_type = dtype_mapping.get(str(dtype), 'STRING')
#             columns_def.append(f"{col} {odps_type}")
        
#         create_table_sql = f"""
#         CREATE TABLE IF NOT EXISTS {table_name} (
#             {',\n            '.join(columns_def)}
#         )
#         PARTITIONED BY (pt STRING)
#         STORED AS ALIORC
#         TBLPROPERTIES ('transactional'='true')
#         """
        
#         print("Generated CREATE TABLE SQL:")
#         print(create_table_sql)
        
#         # Execute the CREATE TABLE statement
#         odps_conn.execute_sql(create_table_sql)
#         print(f"✓ Table '{table_name}' created successfully from DataFrame schema!")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error creating table from DataFrame schema: {e}")
#         return False

# def upload_to_odps(odps_conn, df, table_name, partition_spec, overwrite=True):
#     """
#     Uploads a pandas DataFrame to a partitioned MaxCompute table.
#     Creates the table if it doesn't exist.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         df: The pandas DataFrame to upload.
#         table_name: The name of the target ODPS table.
#         partition_spec: The partition specification string (e.g., "pt='20250801'").
#         overwrite: If True, overwrite the partition if it exists.
#     """
#     if df.empty:
#         print("DataFrame is empty. Nothing to upload.")
#         return

#     try:
#         # First, ensure the table exists
#         if not create_target_table_if_not_exists(odps_conn, table_name):
#             print(f"✗ Failed to create or verify table '{table_name}'. Upload aborted.")
#             return
        
#         print(f"💾 Writing {len(df)} rows to ODPS table '{table_name}' with partition: {partition_spec}")
        
#         # Convert DataFrame columns to match expected types
#         df_upload = df.copy()
        
#         # Handle datetime columns - ensure they're in the right format and remove timezone info
#         datetime_columns = ['submit_time', 'created_time', 'updated_time', 'delete_time', 'migration_time']
#         for col in datetime_columns:
#             if col in df_upload.columns:
#                 # Convert to datetime and remove timezone information to avoid PyArrow issues
#                 df_upload[col] = pd.to_datetime(df_upload[col], errors='coerce')
#                 if df_upload[col].dt.tz is not None:
#                     # Convert timezone-aware datetime to naive datetime (UTC)
#                     df_upload[col] = df_upload[col].dt.tz_convert('UTC').dt.tz_localize(None)
#                 # Ensure datetime format is compatible with MaxCompute
#                 df_upload[col] = df_upload[col].dt.strftime('%Y-%m-%d %H:%M:%S')
#                 df_upload[col] = pd.to_datetime(df_upload[col], errors='coerce')
        
#         # Handle integer columns
#         int_columns = ['have_phone_number', 'whatsapp', 'fixed_or_not', 'is_delete', 'version', 'history_flag']
#         for col in int_columns:
#             if col in df_upload.columns:
#                 df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0).astype('Int64')
        
#         # Ensure all string columns are properly formatted
#         string_columns = ['id', 'customer_id', 'phone_number', 'app_version', 'feedback_type', 
#                          'description', 'mobile_model', 'notes', 'created_by', 'updated_by', 
#                          'delete_by', 'remark', 'ext', 'transfer_work_order', 'customer_email', 
#                          'customer_name', 'predicted_type', 'predicted_explanation', 'work_order', 
#                          'category', 'work_priority']
#         for col in string_columns:
#             if col in df_upload.columns:
#                 df_upload[col] = df_upload[col].astype(str).fillna('')
        
#         # The write_table method handles creating the partition if it doesn't exist
#         # and writing the DataFrame data directly.
#         odps_conn.write_table(
#             table_name, 
#             df_upload, 
#             partition=partition_spec, 
#             overwrite=overwrite, 
#             create_partition=True
#         )
#         print("✓ Upload successful!")
        
#     except Exception as e:
#         print(f"✗ Error uploading data to MaxCompute: {e}")
#         # Print more detailed error information
#         import traceback
#         print(f"Detailed error: {traceback.format_exc()}")

# # ================================
# # HELPER FUNCTIONS
# # ================================

# def load_model_and_tokenizer(model_path):
#     """Loads a Hugging Face model and tokenizer from a specified path."""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         print(f"✓ Model loaded from {model_path}")
#         return model, tokenizer
#     except Exception as e:
#         print(f"✗ Error loading model from {model_path}: {e}")
#         return None, None

# def load_json_mapping(file_path):
#     """Loads a JSON mapping file."""
#     try:
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"✗ Error loading JSON from {file_path}: {e}")
#         return {}

# def predict_batch(model, tokenizer, texts, batch_size=32):
#     """Performs batch prediction on a list of texts."""
#     model.eval()
#     all_preds = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size), desc="🤖 Predicting"):
#             batch_texts = texts[i:i + batch_size]
#             inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#             outputs = model(**inputs)
#             preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#             all_preds.extend(preds)
#     return all_preds

# def map_predictions_to_labels(predictions, config_path):
#     """Maps prediction IDs to their corresponding string labels."""
#     try:
#         with open(config_path) as f:
#             config = json.load(f)
#         id2label = config['id2label']
#         return [id2label.get(str(p), "UNKNOWN") for p in predictions]
#     except Exception as e:
#         print(f"✗ Error mapping predictions: {e}")
#         return ["ERROR"] * len(predictions)

# # ================================
# # MAIN EXECUTION
# # ================================

# def main():
#     """Main function to execute the entire pipeline."""
#     print("🚀 Starting the feedback processing pipeline...")

#     # --- 1. Fetch Data ---
#     odps_connection = create_odps_connection()
#     if not odps_connection:
#         return
#     df = fetch_data_from_odps(odps_connection, query=SOURCE_QUERY)
#     if df is None or df.empty:
#         print("✗ No data fetched or dataframe is empty. Exiting.")
#         return
#     print(f"✓ Fetched {len(df)} records from ODPS.")

#     # --- 2. Load Models & Mappings ---
#     print("\n🧠 Loading models, tokenizers, and mappings...")
#     model_type, tokenizer_type = load_model_and_tokenizer(MODEL_PATHS["type"])
#     model_explanation, tokenizer_explanation = load_model_and_tokenizer(MODEL_PATHS["explanation"])
#     explanation_to_work_order = load_json_mapping(MAPPING_PATHS["explanation_to_work_order"])
#     work_order_to_category = load_json_mapping(MAPPING_PATHS["work_order_to_category"])
#     work_order_to_priority = load_json_mapping(MAPPING_PATHS["work_order_to_work_priority"])

#     if not all([model_type, model_explanation, explanation_to_work_order, work_order_to_category, work_order_to_priority]):
#         print("✗ Failed to load one or more models or mappings. Exiting.")
#         return

#     # --- 3. Preprocess and Predict ---
#     descriptions = df[DESCRIPTION_COLUMN].tolist()

#     # Predict 'Type'
#     predicted_type_ids = predict_batch(model_type, tokenizer_type, descriptions)
#     df['predicted_type'] = map_predictions_to_labels(predicted_type_ids, MAPPING_PATHS["type"])

#     # Predict 'Explanation'
#     predicted_explanation_ids = predict_batch(model_explanation, tokenizer_explanation, descriptions)
#     df['predicted_explanation'] = map_predictions_to_labels(predicted_explanation_ids, MAPPING_PATHS["explanation"])

#     # --- 4. Map Predictions to Final Categories ---
#     print("\n🔄 Mapping predictions to final work order categories...")
#     df['work_order'] = df['predicted_explanation'].map(explanation_to_work_order.get)
#     df['category'] = df['work_order'].map(work_order_to_category.get)
#     df['work_priority'] = df['work_order'].map(work_order_to_priority.get)

#     # --- 5. Prepare and Upload Results ---
#     # Define the columns you want to upload to the new table.
#     # This includes all original columns plus the new predicted columns.
#     columns_to_upload = [
#         # Original columns from the source query
#         'id', 
#         'customer_id', 
#         'phone_number', 
#         'have_phone_number', 
#         'app_version',
#         'feedback_type', 
#         'description', 
#         'mobile_model', 
#         'whatsapp', 
#         'submit_time',
#         'fixed_or_not', 
#         'notes', 
#         'created_time', 
#         'created_by', 
#         'updated_time',
#         'updated_by', 
#         'delete_time', 
#         'delete_by', 
#         'is_delete', 
#         'version',
#         'remark', 
#         'ext', 
#         'history_flag', 
#         'migration_time', 
#         'transfer_work_order',
#         'customer_email', 
#         'customer_name',
#         # New predicted columns added by the ML pipeline
#         'predicted_type', 
#         'predicted_explanation', 
#         'work_order', 
#         'category', 
#         'work_priority'
#     ]
    
#     # Ensure all selected columns exist in the DataFrame before uploading
#     df_to_upload = df[[col for col in columns_to_upload if col in df.columns]].copy()

#     # Generate the dynamic partition string using today's date (or yesterday's if matching the query logic)
#     # The query fetches for DATEADD(..., -1, ...), so we use yesterday's date for consistency.
#     partition_date = datetime.now().strftime('%Y%m%d')
#     dynamic_partition = OUTPUT_PARTITION_TEMPLATE.format(date=partition_date)
    
#     print("\n✨ Final Processed Data (Top 5 rows):")
#     print(df_to_upload.head())

#     # Upload the processed DataFrame to MaxCompute
#     upload_to_odps(odps_connection, df_to_upload, OUTPUT_TABLE_NAME, dynamic_partition)
    
#     print("\n✅ Pipeline finished successfully!")

# if __name__ == '__main__':
#     main()


# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import json
# import os
# from tqdm import tqdm
# from dotenv import load_dotenv
# from odps import ODPS
# from odps.df import DataFrame as ODPSDataFrame
# from datetime import datetime
# import warnings

# warnings.filterwarnings('ignore')

# # ================================
# # CONFIGURATION
# # ================================

# load_dotenv()

# # --- MaxCompute/ODPS Configuration ---
# ODPS_CONFIG = {
#     "access_id": os.getenv("ODPS_ACCESS_ID"),
#     "secret_access_key": os.getenv("ODPS_SECRET_ACCESS_KEY"),
#     "project": os.getenv("ODPS_PROJECT"),
#     "endpoint": os.getenv("ODPS_ENDPOINT")
# }

# # --- Source Table/Query Configuration ---
# SOURCE_QUERY = """
# SELECT
#     id, customer_id, phone_number, have_phone_number, app_version,
#     feedback_type, description, mobile_model, whatsapp, submit_time,
#     fixed_or_not, notes, created_time, created_by, updated_time,
#     updated_by, delete_time, delete_by, is_delete, version,
#     remark, ext, history_flag, migration_time, transfer_work_order,
#     customer_email, customer_name
# FROM kredi_dwd.dwd_customer_support_customer_feedback_df
# WHERE pt = MAX_PT('kredi_dwd.dwd_customer_support_customer_feedback_df')
#     AND to_date(submit_time) = DATEADD(to_date(GETDATE()), -1, 'dd')
#     AND fixed_or_not = 0
# ORDER BY submit_time DESC
# """

# # --- Output Table Configuration ---
# # The target table should exist or be created with a schema that matches the DataFrame.
# OUTPUT_TABLE_NAME = "dwd_customer_support_customer_feedback_hftransformer_df"
# # The '${bizdate}' is a DataWorks parameter. We'll generate this dynamically.
# OUTPUT_PARTITION_TEMPLATE = "pt='{date}'" 

# # --- Column Name ---
# DESCRIPTION_COLUMN = "description"

# # --- Model & Mapping Paths ---
# MODEL_PATHS = {
#     "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
#     "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
# }

# MAPPING_PATHS = {
#     "type": os.path.join(MODEL_PATHS["type"], 'config.json'),
#     "explanation": os.path.join(MODEL_PATHS["explanation"], 'config.json'),
#     "explanation_to_work_order": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\explanation_to_work_order.json",
#     "work_order_to_category": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_category.json",
#     "work_order_to_work_priority": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_work_priority.json",
# }

# # ================================
# # MAXCOMPUTE/ODPS FUNCTIONS
# # ================================

# def create_odps_connection():
#     """Creates and returns a MaxCompute ODPS connection object."""
#     try:
#         odps = ODPS(
#             access_id=ODPS_CONFIG["access_id"],
#             secret_access_key=ODPS_CONFIG["secret_access_key"],
#             project=ODPS_CONFIG["project"],
#             endpoint=ODPS_CONFIG["endpoint"]
#         )
#         print(f"✓ Connected to MaxCompute project: {ODPS_CONFIG['project']}")
#         return odps
#     except Exception as e:
#         print(f"✗ Error connecting to MaxCompute: {e}")
#         return None

# def fetch_data_from_odps(odps, query=None):
#     """Fetches data from MaxCompute by executing a query."""
#     if not odps:
#         return None
#     try:
#         print("📊 Executing ODPS SQL query...")
#         with odps.execute_sql(query).open_reader() as reader:
#             df = reader.to_pandas()
#             return df
#     except Exception as e:
#         print(f"✗ Error fetching data from MaxCompute: {e}")
#         return None

# def create_target_table_if_not_exists(odps_conn, table_name):
#     """
#     Creates the target table if it doesn't exist.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         table_name: The name of the target table to create.
#     """
#     try:
#         # Check if table exists
#         if odps_conn.exist_table(table_name):
#             print(f"✓ Table '{table_name}' already exists.")
#             return True
        
#         print(f"📋 Creating table '{table_name}'...")
        
#         # Define the table schema - adjust data types as needed
#         create_table_sql = f"""
#         CREATE TABLE IF NOT EXISTS {table_name} (
#             id STRING,
#             customer_id STRING,
#             phone_number STRING,
#             have_phone_number BIGINT,
#             app_version STRING,
#             feedback_type STRING,
#             description STRING,
#             mobile_model STRING,
#             whatsapp BIGINT,
#             submit_time DATETIME,
#             fixed_or_not BIGINT,
#             notes STRING,
#             created_time DATETIME,
#             created_by STRING,
#             updated_time DATETIME,
#             updated_by STRING,
#             delete_time DATETIME,
#             delete_by STRING,
#             is_delete BIGINT,
#             version BIGINT,
#             remark STRING,
#             ext STRING,
#             history_flag BIGINT,
#             migration_time DATETIME,
#             transfer_work_order STRING,
#             customer_email STRING,
#             customer_name STRING,
#             predicted_type STRING,
#             predicted_explanation STRING,
#             work_order STRING,
#             category STRING,
#             work_priority STRING
#         )
#         PARTITIONED BY (pt STRING)
#         STORED AS ALIORC
#         TBLPROPERTIES ('transactional'='true')
#         """
        
#         # Execute the CREATE TABLE statement
#         odps_conn.execute_sql(create_table_sql)
#         print(f"✓ Table '{table_name}' created successfully!")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error creating table '{table_name}': {e}")
#         return False

# def create_table_from_dataframe(odps_conn, df, table_name):
#     """
#     Alternative method: Create table schema based on DataFrame dtypes.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         df: The pandas DataFrame to analyze for schema.
#         table_name: The name of the target table to create.
#     """
#     try:
#         if odps_conn.exist_table(table_name):
#             print(f"✓ Table '{table_name}' already exists.")
#             return True
        
#         print(f"📋 Creating table '{table_name}' from DataFrame schema...")
        
#         # Map pandas dtypes to MaxCompute types
#         dtype_mapping = {
#             'object': 'STRING',
#             'int64': 'BIGINT',
#             'int32': 'BIGINT',
#             'float64': 'DOUBLE',
#             'float32': 'DOUBLE',
#             'bool': 'BOOLEAN',
#             'datetime64[ns]': 'DATETIME',
#             'Int64': 'BIGINT'  # pandas nullable integer
#         }
        
#         columns_def = []
#         for col, dtype in df.dtypes.items():
#             odps_type = dtype_mapping.get(str(dtype), 'STRING')
#             columns_def.append(f"{col} {odps_type}")
        
#         create_table_sql = f"""
#         CREATE TABLE IF NOT EXISTS {table_name} (
#             {',\n            '.join(columns_def)}
#         )
#         PARTITIONED BY (pt STRING)
#         STORED AS ALIORC
#         TBLPROPERTIES ('transactional'='true')
#         """
        
#         print("Generated CREATE TABLE SQL:")
#         print(create_table_sql)
        
#         # Execute the CREATE TABLE statement
#         odps_conn.execute_sql(create_table_sql)
#         print(f"✓ Table '{table_name}' created successfully from DataFrame schema!")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error creating table from DataFrame schema: {e}")
#         return False

# def upload_to_odps(odps_conn, df, table_name, partition_spec, overwrite=True):
#     """
#     Uploads a pandas DataFrame to a partitioned MaxCompute table.
#     Creates the table if it doesn't exist.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         df: The pandas DataFrame to upload.
#         table_name: The name of the target ODPS table.
#         partition_spec: The partition specification string (e.g., "pt='20250801'").
#         overwrite: If True, overwrite the partition if it exists.
#     """
#     if df.empty:
#         print("DataFrame is empty. Nothing to upload.")
#         return

#     try:
#         # First, ensure the table exists
#         if not create_target_table_if_not_exists(odps_conn, table_name):
#             print(f"✗ Failed to create or verify table '{table_name}'. Upload aborted.")
#             return
        
#         print(f"💾 Writing {len(df)} rows to ODPS table '{table_name}' with partition: {partition_spec}")
        
#         # Convert DataFrame columns to match expected types
#         df_upload = df.copy()
        
#         # Handle datetime columns - ensure they're in the right format
#         datetime_columns = ['submit_time', 'created_time', 'updated_time', 'delete_time', 'migration_time']
#         for col in datetime_columns:
#             if col in df_upload.columns:
#                 df_upload[col] = pd.to_datetime(df_upload[col], errors='coerce')
        
#         # Handle integer columns
#         int_columns = ['have_phone_number', 'whatsapp', 'fixed_or_not', 'is_delete', 'version', 'history_flag']
#         for col in int_columns:
#             if col in df_upload.columns:
#                 df_upload[col] = pd.to_numeric(df_upload[col], errors='coerce').fillna(0).astype('Int64')
        
#         # Ensure all string columns are properly formatted
#         string_columns = ['id', 'customer_id', 'phone_number', 'app_version', 'feedback_type', 
#                          'description', 'mobile_model', 'notes', 'created_by', 'updated_by', 
#                          'delete_by', 'remark', 'ext', 'transfer_work_order', 'customer_email', 
#                          'customer_name', 'predicted_type', 'predicted_explanation', 'work_order', 
#                          'category', 'work_priority']
#         for col in string_columns:
#             if col in df_upload.columns:
#                 df_upload[col] = df_upload[col].astype(str).fillna('')
        
#         # The write_table method handles creating the partition if it doesn't exist
#         # and writing the DataFrame data directly.
#         odps_conn.write_table(
#             table_name, 
#             df_upload, 
#             partition=partition_spec, 
#             overwrite=overwrite, 
#             create_partition=True
#         )
#         print("✓ Upload successful!")
        
#     except Exception as e:
#         print(f"✗ Error uploading data to MaxCompute: {e}")
#         # Print more detailed error information
#         import traceback
#         print(f"Detailed error: {traceback.format_exc()}")

# # ================================
# # HELPER FUNCTIONS
# # ================================

# def load_model_and_tokenizer(model_path):
#     """Loads a Hugging Face model and tokenizer from a specified path."""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         print(f"✓ Model loaded from {model_path}")
#         return model, tokenizer
#     except Exception as e:
#         print(f"✗ Error loading model from {model_path}: {e}")
#         return None, None

# def load_json_mapping(file_path):
#     """Loads a JSON mapping file."""
#     try:
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"✗ Error loading JSON from {file_path}: {e}")
#         return {}

# def predict_batch(model, tokenizer, texts, batch_size=32):
#     """Performs batch prediction on a list of texts."""
#     model.eval()
#     all_preds = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size), desc="🤖 Predicting"):
#             batch_texts = texts[i:i + batch_size]
#             inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#             outputs = model(**inputs)
#             preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#             all_preds.extend(preds)
#     return all_preds

# def map_predictions_to_labels(predictions, config_path):
#     """Maps prediction IDs to their corresponding string labels."""
#     try:
#         with open(config_path) as f:
#             config = json.load(f)
#         id2label = config['id2label']
#         return [id2label.get(str(p), "UNKNOWN") for p in predictions]
#     except Exception as e:
#         print(f"✗ Error mapping predictions: {e}")
#         return ["ERROR"] * len(predictions)

# # ================================
# # MAIN EXECUTION
# # ================================

# def main():
#     """Main function to execute the entire pipeline."""
#     print("🚀 Starting the feedback processing pipeline...")

#     # --- 1. Fetch Data ---
#     odps_connection = create_odps_connection()
#     if not odps_connection:
#         return
#     df = fetch_data_from_odps(odps_connection, query=SOURCE_QUERY)
#     if df is None or df.empty:
#         print("✗ No data fetched or dataframe is empty. Exiting.")
#         return
#     print(f"✓ Fetched {len(df)} records from ODPS.")

#     # --- 2. Load Models & Mappings ---
#     print("\n🧠 Loading models, tokenizers, and mappings...")
#     model_type, tokenizer_type = load_model_and_tokenizer(MODEL_PATHS["type"])
#     model_explanation, tokenizer_explanation = load_model_and_tokenizer(MODEL_PATHS["explanation"])
#     explanation_to_work_order = load_json_mapping(MAPPING_PATHS["explanation_to_work_order"])
#     work_order_to_category = load_json_mapping(MAPPING_PATHS["work_order_to_category"])
#     work_order_to_priority = load_json_mapping(MAPPING_PATHS["work_order_to_work_priority"])

#     if not all([model_type, model_explanation, explanation_to_work_order, work_order_to_category, work_order_to_priority]):
#         print("✗ Failed to load one or more models or mappings. Exiting.")
#         return

#     # --- 3. Preprocess and Predict ---
#     descriptions = df[DESCRIPTION_COLUMN].tolist()

#     # Predict 'Type'
#     predicted_type_ids = predict_batch(model_type, tokenizer_type, descriptions)
#     df['predicted_type'] = map_predictions_to_labels(predicted_type_ids, MAPPING_PATHS["type"])

#     # Predict 'Explanation'
#     predicted_explanation_ids = predict_batch(model_explanation, tokenizer_explanation, descriptions)
#     df['predicted_explanation'] = map_predictions_to_labels(predicted_explanation_ids, MAPPING_PATHS["explanation"])

#     # --- 4. Map Predictions to Final Categories ---
#     print("\n🔄 Mapping predictions to final work order categories...")
#     df['work_order'] = df['predicted_explanation'].map(explanation_to_work_order.get)
#     df['category'] = df['work_order'].map(work_order_to_category.get)
#     df['work_priority'] = df['work_order'].map(work_order_to_priority.get)

#     # --- 5. Prepare and Upload Results ---
#     # Define the columns you want to upload to the new table.
#     # This includes all original columns plus the new predicted columns.
#     columns_to_upload = [
#         # Original columns from the source query
#         'id', 
#         'customer_id', 
#         'phone_number', 
#         'have_phone_number', 
#         'app_version',
#         'feedback_type', 
#         'description', 
#         'mobile_model', 
#         'whatsapp', 
#         'submit_time',
#         'fixed_or_not', 
#         'notes', 
#         'created_time', 
#         'created_by', 
#         'updated_time',
#         'updated_by', 
#         'delete_time', 
#         'delete_by', 
#         'is_delete', 
#         'version',
#         'remark', 
#         'ext', 
#         'history_flag', 
#         'migration_time', 
#         'transfer_work_order',
#         'customer_email', 
#         'customer_name',
#         # New predicted columns added by the ML pipeline
#         'predicted_type', 
#         'predicted_explanation', 
#         'work_order', 
#         'category', 
#         'work_priority'
#     ]
    
#     # Ensure all selected columns exist in the DataFrame before uploading
#     df_to_upload = df[[col for col in columns_to_upload if col in df.columns]].copy()

#     # Generate the dynamic partition string using today's date (or yesterday's if matching the query logic)
#     # The query fetches for DATEADD(..., -1, ...), so we use yesterday's date for consistency.
#     partition_date = (datetime.now() - pd.Timedelta(days=0)).strftime('%Y%m%d')
#     dynamic_partition = OUTPUT_PARTITION_TEMPLATE.format(date=partition_date)
    
#     print("\n✨ Final Processed Data (Top 5 rows):")
#     print(df_to_upload.head())

#     # Upload the processed DataFrame to MaxCompute
#     upload_to_odps(odps_connection, df_to_upload, OUTPUT_TABLE_NAME, dynamic_partition)
    
#     print("\n✅ Pipeline finished successfully!")

# if __name__ == '__main__':
#     main()


# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import json
# import os
# from tqdm import tqdm
# from dotenv import load_dotenv
# from odps import ODPS
# from odps.df import DataFrame as ODPSDataFrame
# from datetime import datetime
# import warnings

# warnings.filterwarnings('ignore')

# # ================================
# # CONFIGURATION
# # ================================

# load_dotenv()

# # --- MaxCompute/ODPS Configuration ---
# ODPS_CONFIG = {
#     "access_id": os.getenv("ODPS_ACCESS_ID"),
#     "secret_access_key": os.getenv("ODPS_SECRET_ACCESS_KEY"),
#     "project": os.getenv("ODPS_PROJECT"),
#     "endpoint": os.getenv("ODPS_ENDPOINT")
# }

# # --- Source Table/Query Configuration ---
# SOURCE_QUERY = """
# SELECT
#     id, customer_id, phone_number, have_phone_number, app_version,
#     feedback_type, description, mobile_model, whatsapp, submit_time,
#     fixed_or_not, notes, created_time, created_by, updated_time,
#     updated_by, delete_time, delete_by, is_delete, version,
#     remark, ext, history_flag, migration_time, transfer_work_order,
#     customer_email, customer_name
# FROM kredi_dwd.dwd_customer_support_customer_feedback_df
# WHERE pt = MAX_PT('kredi_dwd.dwd_customer_support_customer_feedback_df')
#     AND to_date(submit_time) = DATEADD(to_date(GETDATE()), -1, 'dd')
#     AND fixed_or_not = 0
# ORDER BY submit_time DESC
# """

# # --- Output Table Configuration ---
# # The target table should exist or be created with a schema that matches the DataFrame.
# OUTPUT_TABLE_NAME = "dwd_customer_support_customer_feedback_hftransformer_df"
# # The '${bizdate}' is a DataWorks parameter. We'll generate this dynamically.
# OUTPUT_PARTITION_TEMPLATE = "pt='{date}'" 

# # --- Column Name ---
# DESCRIPTION_COLUMN = "description"

# # --- Model & Mapping Paths ---
# MODEL_PATHS = {
#     "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
#     "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
# }

# MAPPING_PATHS = {
#     "type": os.path.join(MODEL_PATHS["type"], 'config.json'),
#     "explanation": os.path.join(MODEL_PATHS["explanation"], 'config.json'),
#     "explanation_to_work_order": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\explanation_to_work_order.json",
#     "work_order_to_category": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_category.json",
#     "work_order_to_work_priority": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_work_priority.json",
# }

# # ================================
# # MAXCOMPUTE/ODPS FUNCTIONS
# # ================================

# def create_odps_connection():
#     """Creates and returns a MaxCompute ODPS connection object."""
#     try:
#         odps = ODPS(
#             access_id=ODPS_CONFIG["access_id"],
#             secret_access_key=ODPS_CONFIG["secret_access_key"],
#             project=ODPS_CONFIG["project"],
#             endpoint=ODPS_CONFIG["endpoint"]
#         )
#         print(f"✓ Connected to MaxCompute project: {ODPS_CONFIG['project']}")
#         return odps
#     except Exception as e:
#         print(f"✗ Error connecting to MaxCompute: {e}")
#         return None

# def fetch_data_from_odps(odps, query=None):
#     """Fetches data from MaxCompute by executing a query."""
#     if not odps:
#         return None
#     try:
#         print("📊 Executing ODPS SQL query...")
#         with odps.execute_sql(query).open_reader() as reader:
#             df = reader.to_pandas()
#             return df
#     except Exception as e:
#         print(f"✗ Error fetching data from MaxCompute: {e}")
#         return None

# def upload_to_odps(odps_conn, df, table_name, partition_spec, overwrite=True):
#     """
#     Uploads a pandas DataFrame to a partitioned MaxCompute table.
    
#     Args:
#         odps_conn: The ODPS connection object.
#         df: The pandas DataFrame to upload.
#         table_name: The name of the target ODPS table.
#         partition_spec: The partition specification string (e.g., "pt='20250801'").
#         overwrite: If True, overwrite the partition if it exists.
#     """
#     if df.empty:
#         print("DataFrame is empty. Nothing to upload.")
#         return

#     try:
#         print(f"💾 Writing {len(df)} rows to ODPS table '{table_name}' with partition: {partition_spec}")
#         # The write_table method handles creating the partition if it doesn't exist
#         # and writing the DataFrame data directly.
#         odps_conn.write_table(
#             table_name, 
#             df, 
#             partition=partition_spec, 
#             overwrite=overwrite, 
#             create_partition=True
#         )
#         print("✓ Upload successful!")
#     except Exception as e:
#         print(f"✗ Error uploading data to MaxCompute: {e}")

# # ================================
# # HELPER FUNCTIONS
# # ================================

# def load_model_and_tokenizer(model_path):
#     """Loads a Hugging Face model and tokenizer from a specified path."""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         print(f"✓ Model loaded from {model_path}")
#         return model, tokenizer
#     except Exception as e:
#         print(f"✗ Error loading model from {model_path}: {e}")
#         return None, None

# def load_json_mapping(file_path):
#     """Loads a JSON mapping file."""
#     try:
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"✗ Error loading JSON from {file_path}: {e}")
#         return {}

# def predict_batch(model, tokenizer, texts, batch_size=32):
#     """Performs batch prediction on a list of texts."""
#     model.eval()
#     all_preds = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size), desc="🤖 Predicting"):
#             batch_texts = texts[i:i + batch_size]
#             inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#             outputs = model(**inputs)
#             preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#             all_preds.extend(preds)
#     return all_preds

# def map_predictions_to_labels(predictions, config_path):
#     """Maps prediction IDs to their corresponding string labels."""
#     try:
#         with open(config_path) as f:
#             config = json.load(f)
#         id2label = config['id2label']
#         return [id2label.get(str(p), "UNKNOWN") for p in predictions]
#     except Exception as e:
#         print(f"✗ Error mapping predictions: {e}")
#         return ["ERROR"] * len(predictions)

# # ================================
# # MAIN EXECUTION
# # ================================

# def main():
#     """Main function to execute the entire pipeline."""
#     print("🚀 Starting the feedback processing pipeline...")

#     # --- 1. Fetch Data ---
#     odps_connection = create_odps_connection()
#     if not odps_connection:
#         return
#     df = fetch_data_from_odps(odps_connection, query=SOURCE_QUERY)
#     if df is None or df.empty:
#         print("✗ No data fetched or dataframe is empty. Exiting.")
#         return
#     print(f"✓ Fetched {len(df)} records from ODPS.")

#     # --- 2. Load Models & Mappings ---
#     print("\n🧠 Loading models, tokenizers, and mappings...")
#     model_type, tokenizer_type = load_model_and_tokenizer(MODEL_PATHS["type"])
#     model_explanation, tokenizer_explanation = load_model_and_tokenizer(MODEL_PATHS["explanation"])
#     explanation_to_work_order = load_json_mapping(MAPPING_PATHS["explanation_to_work_order"])
#     work_order_to_category = load_json_mapping(MAPPING_PATHS["work_order_to_category"])
#     work_order_to_priority = load_json_mapping(MAPPING_PATHS["work_order_to_work_priority"])

#     if not all([model_type, model_explanation, explanation_to_work_order, work_order_to_category, work_order_to_priority]):
#         print("✗ Failed to load one or more models or mappings. Exiting.")
#         return

#     # --- 3. Preprocess and Predict ---
#     descriptions = df[DESCRIPTION_COLUMN].tolist()

#     # Predict 'Type'
#     predicted_type_ids = predict_batch(model_type, tokenizer_type, descriptions)
#     df['predicted_type'] = map_predictions_to_labels(predicted_type_ids, MAPPING_PATHS["type"])

#     # Predict 'Explanation'
#     predicted_explanation_ids = predict_batch(model_explanation, tokenizer_explanation, descriptions)
#     df['predicted_explanation'] = map_predictions_to_labels(predicted_explanation_ids, MAPPING_PATHS["explanation"])

#     # --- 4. Map Predictions to Final Categories ---
#     print("\n🔄 Mapping predictions to final work order categories...")
#     df['work_order'] = df['predicted_explanation'].map(explanation_to_work_order.get)
#     df['category'] = df['work_order'].map(work_order_to_category.get)
#     df['work_priority'] = df['work_order'].map(work_order_to_priority.get)

#     # --- 5. Prepare and Upload Results ---
#     # Define the columns you want to upload to the new table.
#     # This ensures the schema is clean and intentional.
#     columns_to_upload = [
#         # Original columns from the source query
#         'id', 
#         'customer_id', 
#         'phone_number', 
#         'have_phone_number', 
#         'app_version',
#         'feedback_type', 
#         'description', 
#         'mobile_model', 
#         'whatsapp', 
#         'submit_time',
#         'fixed_or_not', 
#         'notes', 
#         'created_time', 
#         'created_by', 
#         'updated_time',
#         'updated_by', 
#         'delete_time', 
#         'delete_by', 
#         'is_delete', 
#         'version',
#         'remark', 
#         'ext', 
#         'history_flag', 
#         'migration_time', 
#         'transfer_work_order',
#         'customer_email', 
#         'customer_name',
#         # New predicted columns added by the ML pipeline
#         'predicted_type', 
#         'predicted_explanation', 
#         'work_order', 
#         'category', 
#         'work_priority'
#     ]
#     # Ensure all selected columns exist in the DataFrame before uploading
#     df_to_upload = df[[col for col in columns_to_upload if col in df.columns]].copy()

#     # Generate the dynamic partition string using today's date (or yesterday's if matching the query logic)
#     # The query fetches for DATEADD(..., -1, ...), so we use yesterday's date for consistency.
#     partition_date = (datetime.now() - pd.Timedelta(days=0)).strftime('%Y%m%d')
#     dynamic_partition = OUTPUT_PARTITION_TEMPLATE.format(date=partition_date)
    
#     print("\n✨ Final Processed Data (Top 5 rows):")
#     print(df_to_upload.head())

#     # Upload the processed DataFrame to MaxCompute
#     upload_to_odps(odps_connection, df_to_upload, OUTPUT_TABLE_NAME, dynamic_partition)
    
#     print("\n✅ Pipeline finished successfully!")

# if __name__ == '__main__':
#     main()


# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import json
# import os
# from tqdm import tqdm
# from dotenv import load_dotenv
# from odps import ODPS
# from odps.df import DataFrame as ODPSDataFrame
# from datetime import datetime
# import warnings

# warnings.filterwarnings('ignore')

# # ================================
# # CONFIGURATION
# # ================================

# load_dotenv()

# # --- MaxCompute/ODPS Configuration ---
# ODPS_CONFIG = {
#     "access_id": os.getenv("ODPS_ACCESS_ID"),
#     "secret_access_key": os.getenv("ODPS_SECRET_ACCESS_KEY"),
#     "project": os.getenv("ODPS_PROJECT"),
#     "endpoint": os.getenv("ODPS_ENDPOINT")
# }

# # --- Source Table/Query Configuration ---
# SOURCE_QUERY = """
# SELECT
#     id, customer_id, phone_number, have_phone_number, app_version,
#     feedback_type, description, mobile_model, whatsapp, submit_time,
#     fixed_or_not, notes, created_time, created_by, updated_time,
#     updated_by, delete_time, delete_by, is_delete, version,
#     remark, ext, history_flag, migration_time, transfer_work_order,
#     customer_email, customer_name
# FROM kredi_dwd.dwd_customer_support_customer_feedback_df
# WHERE pt = MAX_PT('kredi_dwd.dwd_customer_support_customer_feedback_df')
#     AND to_date(submit_time) = DATEADD(to_date(GETDATE()), -1, 'dd')
#     AND fixed_or_not = 0
# ORDER BY submit_time DESC
# """

# # --- Output Table Configuration ---
# OUTPUT_TABLE_NAME = "dwd_customer_support_customer_feedback_hftransformer_df"
# OUTPUT_PARTITION = "pt='${bizdate}'"

# # --- Column Name ---
# DESCRIPTION_COLUMN = "description"

# # --- Model & Mapping Paths ---
# # Adjust these paths to your local environment
# MODEL_PATHS = {
#     "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
#     "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
# }

# MAPPING_PATHS = {
#     "type": os.path.join(MODEL_PATHS["type"], 'config.json'),
#     "explanation": os.path.join(MODEL_PATHS["explanation"], 'config.json'),
#     "explanation_to_work_order": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\explanation_to_work_order.json",
#     "work_order_to_category": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_category.json",
#     "work_order_to_work_priority": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_work_priority.json",
# }

# SLANG_DICT_URL = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

# # ================================
# # MAXCOMPUTE/ODPS FUNCTIONS
# # ================================

# def create_odps_connection():
#     """Creates and returns a MaxCompute ODPS connection object."""
#     try:
#         odps = ODPS(
#             access_id=ODPS_CONFIG["access_id"],
#             secret_access_key=ODPS_CONFIG["secret_access_key"],
#             project=ODPS_CONFIG["project"],
#             endpoint=ODPS_CONFIG["endpoint"]
#         )
#         # odps.list_tables(max_items=1)
#         print(f"✓ Connected to MaxCompute project: {ODPS_CONFIG['project']}")
#         return odps
#     except Exception as e:
#         print(f"✗ Error connecting to MaxCompute: {e}")
#         return None

# def fetch_data_from_odps(odps, query=None):
#     """Fetches data from MaxCompute by executing a query."""
#     if not odps:
#         return None
#     try:
#         print("📊 Executing ODPS SQL query...")
#         with odps.execute_sql(query).open_reader() as reader:
#             df = reader.to_pandas()
#             return df
#     except Exception as e:
#         print(f"✗ Error fetching data from MaxCompute: {e}")
#         return None

# # ================================
# # HELPER FUNCTIONS
# # ================================

# def load_model_and_tokenizer(model_path):
#     """Loads a Hugging Face model and tokenizer from a specified path."""
#     try:
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         model = AutoModelForSequenceClassification.from_pretrained(model_path)
#         print(f"✓ Model loaded from {model_path}")
#         return model, tokenizer
#     except Exception as e:
#         print(f"✗ Error loading model from {model_path}: {e}")
#         return None, None

# def load_json_mapping(file_path):
#     """Loads a JSON mapping file."""
#     try:
#         with open(file_path, 'r') as f:
#             return json.load(f)
#     except Exception as e:
#         print(f"✗ Error loading JSON from {file_path}: {e}")
#         return {}

# def predict_batch(model, tokenizer, texts, batch_size=32):
#     """Performs batch prediction on a list of texts."""
#     model.eval()
#     all_preds = []
#     with torch.no_grad():
#         for i in tqdm(range(0, len(texts), batch_size), desc="🤖 Predicting"):
#             batch_texts = texts[i:i + batch_size]
#             inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
#             outputs = model(**inputs)
#             preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
#             all_preds.extend(preds)
#     return all_preds

# def map_predictions_to_labels(predictions, config_path):
#     """Maps prediction IDs to their corresponding string labels."""
#     try:
#         with open(config_path) as f:
#             config = json.load(f)
#         id2label = config['id2label']
#         return [id2label.get(str(p), "UNKNOWN") for p in predictions]
#     except Exception as e:
#         print(f"✗ Error mapping predictions: {e}")
#         return ["ERROR"] * len(predictions)

# # ================================
# # MAIN EXECUTION
# # ================================

# def main():
#     """Main function to execute the entire pipeline."""
#     print("🚀 Starting the feedback processing pipeline...")

#     # --- 1. Fetch Data ---
#     odps_connection = create_odps_connection()
#     if not odps_connection:
#         return
#     df = fetch_data_from_odps(odps_connection, query=SOURCE_QUERY)
#     if df is None or df.empty:
#         print("✗ No data fetched or dataframe is empty. Exiting.")
#         return
#     print(f"✓ Fetched {len(df)} records from ODPS.")

#     # --- 2. Load Models & Mappings ---
#     print("🧠 Loading models, tokenizers, and mappings...")
#     model_type, tokenizer_type = load_model_and_tokenizer(MODEL_PATHS["type"])
#     model_explanation, tokenizer_explanation = load_model_and_tokenizer(MODEL_PATHS["explanation"])
#     explanation_to_work_order = load_json_mapping(MAPPING_PATHS["explanation_to_work_order"])
#     work_order_to_category = load_json_mapping(MAPPING_PATHS["work_order_to_category"])
#     work_order_to_priority = load_json_mapping(MAPPING_PATHS["work_order_to_work_priority"])

#     if not all([model_type, model_explanation, explanation_to_work_order, work_order_to_category, work_order_to_priority]):
#         print("✗ Failed to load one or more models or mappings. Exiting.")
#         return

#     # --- 3. Preprocess and Predict ---
#     descriptions = df[DESCRIPTION_COLUMN].tolist()

#     # Predict 'Type'
#     predicted_type_ids = predict_batch(model_type, tokenizer_type, descriptions)
#     df['predicted_type'] = map_predictions_to_labels(predicted_type_ids, MAPPING_PATHS["type"])

#     # Predict 'Explanation'
#     predicted_explanation_ids = predict_batch(model_explanation, tokenizer_explanation, descriptions)
#     df['predicted_explanation'] = map_predictions_to_labels(predicted_explanation_ids, MAPPING_PATHS["explanation"])

#     # --- 4. Map Predictions to Final Categories ---
#     print("🔄 Mapping predictions to final work order categories...")
#     df['work_order'] = df['predicted_explanation'].map(explanation_to_work_order.get)
#     df['category'] = df['work_order'].map(work_order_to_category.get)
#     df['work_priority'] = df['work_order'].map(work_order_to_priority.get)

#     # --- 5. Display and Save Results ---
#     print("\n✨ Final Processed Data (Top 5 rows):")
#     print(df[['id', DESCRIPTION_COLUMN, 'predicted_type', 'predicted_explanation', 'work_order', 'category', 'work_priority']].head())

#     # Here you would add the logic to write the DataFrame `df` back to your target ODPS table.
#     # For example:
#     # print(f"💾 Writing results to ODPS table: {OUTPUT_TABLE_NAME} with partition: {OUTPUT_PARTITION}")
#     # odps_connection.write_table(OUTPUT_TABLE_NAME, df, partition=OUTPUT_PARTITION, overwrite=True)

#     print("\n✅ Pipeline finished successfully!")

# if __name__ == '__main__':
#     main()



# import pandas as pd
# import torch
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import re
# import json
# import os
# from tqdm import tqdm
# from dotenv import load_dotenv
# from odps import ODPS
# from odps.df import DataFrame as ODPSDataFrame
# from datetime import datetime
# import warnings
# warnings.filterwarnings('ignore')

# # ================================
# # CONFIGURATION
# # ================================

# load_dotenv()

# # --- MaxCompute/ODPS Configuration ---
# ODPS_CONFIG = {
#     "access_id": os.getenv("ODPS_ACCESS_ID"),
#     "secret_access_key": os.getenv("ODPS_SECRET_ACCESS_KEY"),
#     "project": os.getenv("ODPS_PROJECT"),
#     "endpoint": os.getenv("ODPS_ENDPOINT")
# }

# # --- Source Table/Query Configuration ---
# SOURCE_TABLE = None  
# SOURCE_QUERY = """
# SELECT 
#     id,
#     customer_id,
#     phone_number,
#     have_phone_number,
#     app_version,
#     feedback_type,
#     description,
#     mobile_model,
#     whatsapp,
#     submit_time,
#     fixed_or_not,
#     notes,
#     created_time,
#     created_by,
#     updated_time,
#     updated_by,
#     delete_time,
#     delete_by,
#     is_delete,
#     version,
#     remark,
#     ext,
#     history_flag,
#     migration_time,
#     transfer_work_order,
#     customer_email,
#     customer_name
# FROM kredi_dwd.dwd_customer_support_customer_feedback_df
# WHERE pt = MAX_PT('kredi_dwd.dwd_customer_support_customer_feedback_df')
#     AND to_date(submit_time) = DATEADD(to_date(GETDATE()), -1, 'dd')  -- Yesterday's submissions
#     AND feedback_type = 7  -- Specific feedback type
#     AND fixed_or_not = 0   -- Only unfixed feedback
#     AND description IS NOT NULL 
#     AND description != ''
# ORDER BY submit_time DESC
# -- LIMIT 10  -- Remove limit for production, keep for testing
# """

# # --- Output Table Configuration ---
# OUTPUT_TABLE_NAME = "dwd_customer_support_customer_feedback_hftransformer_df"
# OUTPUT_PARTITION = "pt='${bizdate}'"  

# # --- Column Name ---
# DESCRIPTION_COLUMN = "description"

# # --- Model & Mapping Paths ---
# MODEL_PATHS = {
#     "type": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_type_ft",
#     "explanation": "C:\\Users\\ITN\\Downloads\\models(1)\\models\\model_explanation_ft"
# }

# MAPPING_PATHS = {
#     "explanation_to_work_order": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\explanation_to_work_order.json",
#     "work_order_to_category": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_category.json",
#     "work_order_to_work_priority": "C:\\Users\\ITN\\Documents\\GitHub\\CS-ML-Mapper\\work_order_to_work_priority.json",
# }

# SLANG_DICT_URL = "https://raw.githubusercontent.com/nasalsabila/kamus-alay/master/colloquial-indonesian-lexicon.csv"

# # ================================
# # MAXCOMPUTE/ODPS CONNECTION
# # ================================

# def create_odps_connection():
#     try:
#         odps = ODPS(
#             access_id=ODPS_CONFIG["access_id"],
#             secret_access_key=ODPS_CONFIG["secret_access_key"],
#             project=ODPS_CONFIG["project"],
#             endpoint=ODPS_CONFIG["endpoint"]
#         )
        
#         # Test connection
#         odps.list_tables(max_items=1)
#         print(f"✓ Connected to MaxCompute project: {ODPS_CONFIG['project']}")
#         return odps
#     except Exception as e:
#         print(f"✗ Error connecting to MaxCompute: {e}")
#         return None

# def fetch_data_from_odps(odps, table_name=None, query=None):
#     try:
#         if query:
#             print("📊 Executing ODPS SQL query...")
#             # Execute SQL query
#             with odps.execute_sql(query).open_reader() as reader:
#                 # Convert to pandas DataFrame
#                 data = []
#                 columns = [col.name for col in reader.schema.columns]
                
#                 for record in reader:
#                     row = []
#                     for i, col in enumerate(reader.schema.columns):
#                         row.append(record[i])
#                     data.append(row)
                
#                 df = pd.DataFrame(data, columns=columns)
                
#         elif table_name:
#             print(f"📊 Reading from ODPS table: {table_name}")
#             # Read from table directly
#             table = odps.get_table(table_name)
            
#             # Use DataFrame API for better performance
#             odps_df = ODPSDataFrame(table)
            
#             # Convert to pandas (this downloads all data to local memory)
#             df = odps_df.to_pandas()
            
#         else:
#             raise ValueError("Either table_name or query must be provided")
            
#         print(f"✓ Retrieved {len(df)} rows from MaxCompute")
#         print(f"  Columns: {list(df.columns)}")
#         return df
        
#     except Exception as e:
#         print(f"✗ Error fetching data from ODPS: {e}")
#         return None

# def create_odps_output_table(odps, df, table_name, partition=None):
#     """Create or update ODPS table with predictions."""
#     try:
#         print(f"💾 Preparing to write to ODPS table: {table_name}")
        
#         # Add metadata columns for ML predictions
#         df['prediction_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#         df['model_version'] = "hf_transformers_v1.0"
#         df['processing_date'] = datetime.now().strftime('%Y%m%d')
        
#         # Define table schema based on your original DDL + prediction columns
#         from odps.models import Schema, Column
        
#         # Define the complete schema matching your DDL
#         columns = [
#             # Original table columns
#             Column(name='id', type='bigint', comment='主键Id'),
#             Column(name='customer_id', type='bigint', comment='客户ID'),
#             Column(name='phone_number', type='string', comment='手机号码'),
#             Column(name='have_phone_number', type='bigint', comment='是否有手机号码 0:否 1:是'),
#             Column(name='app_version', type='string', comment='APP版本号'),
#             Column(name='feedback_type', type='string', comment='反馈类型'),
#             Column(name='description', type='string', comment='描述'),
#             Column(name='mobile_model', type='string', comment='手机型号'),
#             Column(name='whatsapp', type='string', comment='WhatsAPP号码'),
#             Column(name='submit_time', type='datetime', comment='反馈时间'),
#             Column(name='fixed_or_not', type='bigint', comment='是否修正 0:未修正，1：已修正'),
#             Column(name='notes', type='string', comment='客服记录'),
#             Column(name='created_time', type='datetime', comment='创建时间'),
#             Column(name='created_by', type='string', comment='创建人'),
#             Column(name='updated_time', type='datetime', comment='更新时间'),
#             Column(name='updated_by', type='string', comment='更新人'),
#             Column(name='delete_time', type='datetime', comment='删除时间'),
#             Column(name='delete_by', type='string', comment='删除人'),
#             Column(name='is_delete', type='bigint', comment='是否删除,0代表未删除,1代表已删除'),
#             Column(name='version', type='bigint', comment='乐观锁版本号'),
#             Column(name='remark', type='string', comment='备注'),
#             Column(name='ext', type='string', comment='扩展字段'),
#             Column(name='history_flag', type='bigint', comment='是否迁移数据，默认进来的数据都是新系统的数据（1-历史数据 2-新数据）'),
#             Column(name='migration_time', type='datetime', comment='迁移时间'),
#             Column(name='transfer_work_order', type='bigint', comment='transfer work order 1:transferred 0:untransferred'),
#             Column(name='customer_email', type='string', comment='user email submit'),
#             Column(name='customer_name', type='string', comment='客户姓名'),
            
#             # ML Prediction columns
#             Column(name='predicted_type', type='string', comment='ML预测的反馈类型'),
#             Column(name='predicted_explanation', type='string', comment='ML预测的问题解释'),
#             Column(name='predicted_work_order_category', type='string', comment='ML预测的工单分类'),
#             Column(name='predicted_category', type='string', comment='ML预测的类别'),
#             Column(name='predicted_work_priority', type='string', comment='ML预测的工作优先级'),
#             Column(name='preprocessed_description', type='string', comment='预处理后的描述文本'),
            
#             # Metadata columns
#             Column(name='prediction_timestamp', type='timestamp', comment='预测时间戳'),
#             Column(name='model_version', type='string', comment='模型版本'),
#             Column(name='processing_date', type='date', comment='处理日期')
#         ]
        
#         # Add partition columns if specified
#         partitions = []
#         if partition:
#             # Parse partition string like "pt='20250731'"
#             if 'pt=' in partition:
#                 partitions.append(Column(name='pt', type='string', comment='数据分区日期'))
        
#         schema = Schema(columns=columns, partitions=partitions)
        
#         # Create or get table
#         if odps.exist_table(table_name):
#             print(f"  Table {table_name} already exists, will append/overwrite partition")
#             table = odps.get_table(table_name)
#         else:
#             print(f"  Creating new table: {table_name}")
#             table = odps.create_table(table_name, schema)
        
#         # Prepare data for writing - ensure all columns are included
#         records = []
#         for _, row in df.iterrows():
#             record = []
#             for col in columns:
#                 if col.name in df.columns:
#                     value = row[col.name]
#                     # Handle different data types and NaN values
#                     if pd.isna(value):
#                         value = None
#                     elif col.type == 'datetime' and value is not None:
#                         # Ensure datetime format is correct
#                         if isinstance(value, str):
#                             try:
#                                 value = pd.to_datetime(value)
#                             except:
#                                 value = None
#                     elif col.type == 'bigint' and value is not None:
#                         # Ensure integer values
#                         try:
#                             value = int(value) if not pd.isna(value) else None
#                         except:
#                             value = None
#                     record.append(value)
#                 else:
#                     # Column not in DataFrame, set to None
#                     record.append(None)
#             records.append(record)
        
#         # Write data to table
#         if partition:
#             # Write to specific partition
#             print(f"  Writing to partition: {partition}")
#             with table.open_writer(partition=partition, create_partition=True) as writer:
#                 writer.write(records)
#         else:
#             # Write to table without partition
#             with table.open_writer() as writer:
#                 writer.write(records)
        
#         print(f"✓ Successfully wrote {len(df)} records to ODPS table: {table_name}")
#         return True
        
#     except Exception as e:
#         print(f"✗ Error writing to ODPS table: {e}")
#         return False

# # ================================
# # SETUP: LOAD MODELS & MAPPINGS
# # ================================
# def load_assets():
#     """Load all Hugging Face models, tokenizers, mappings, and slang dictionary."""
#     print("🧠 Loading models, mappings, and dictionaries...")
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"✓ Using device: {device}")

#     # Load Models and Tokenizers
#     models = {}
#     tokenizers = {}
#     for name, path in MODEL_PATHS.items():
#         if not os.path.isdir(path):
#             print(f"✗ FATAL: Model directory not found at {path}")
#             return None, None, None, None, None
        
#         tokenizers[name] = AutoTokenizer.from_pretrained(path)
#         model = AutoModelForSequenceClassification.from_pretrained(path)
#         model.to(device)
#         model.eval()
#         models[name] = model
#         print(f"✓ Loaded tokenizer and model '{name}'")

#     # Load Mappings
#     mappings = {}
#     for name, path in MAPPING_PATHS.items():
#         try:
#             with open(path, 'r', encoding='utf-8') as f:
#                 mappings[name] = json.load(f)
#             print(f"✓ Loaded mapping '{name}'")
#         except FileNotFoundError:
#             print(f"✗ FATAL: Mapping file not found at {path}")
#             return None, None, None, None, None
            
#     # Load Slang Dictionary
#     try:
#         df_slang = pd.read_csv(SLANG_DICT_URL)
#         slang_dict = dict(zip(df_slang['slang'], df_slang['formal']))
#         print(f"✓ Loaded {len(slang_dict)} slang words")
#     except Exception as e:
#         print(f"⚠ Warning: Could not load slang dictionary: {e}")
#         slang_dict = {}

#     return models, tokenizers, mappings, slang_dict, device

# def text_preprocessing_process(text, slang_dict):
#     """Preprocess text data."""
#     if not isinstance(text, str): 
#         text = str(text)
#     text = text.lower()
#     text = re.sub(r'https?://\S+|www\.\S+', '', text)
#     text = re.sub(r'[-+]?[0-9]+', '', text)
#     text = re.sub(r'[^\w\s]', '', text)
#     text = text.strip()
#     words = text.split()
#     normalized_words = [slang_dict.get(word, word) for word in words]
#     return ' '.join(normalized_words)

# # ================================
# # BATCH PREDICTION
# # ================================

# def predict_with_hf_model(texts, model, tokenizer, device, batch_size=32):
#     """Makes predictions for a list of texts in batches."""
#     model.eval()
#     predictions = []
    
#     for i in tqdm(range(0, len(texts), batch_size), desc=f"Predicting"):
#         batch_texts = texts[i:i + batch_size]
        
#         inputs = tokenizer(
#             batch_texts, 
#             padding=True, 
#             truncation=True, 
#             max_length=512, 
#             return_tensors="pt"
#         )
        
#         inputs = {k: v.to(device) for k, v in inputs.items()}
        
#         with torch.no_grad():
#             outputs = model(**inputs)
        
#         predicted_ids = torch.argmax(outputs.logits, dim=-1)
#         batch_preds = [model.config.id2label[id_] for id_ in predicted_ids.cpu().tolist()]
#         predictions.extend(batch_preds)
        
#     return predictions

# def run_batch_predictions(df, models, tokenizers, mappings, slang_dict, device):
#     """Process DataFrame to add prediction columns."""
#     print("\n🔄 Preprocessing all descriptions...")
    
#     # Handle missing descriptions
#     df[DESCRIPTION_COLUMN] = df[DESCRIPTION_COLUMN].fillna("").astype(str)
#     preprocessed_texts = [text_preprocessing_process(text, slang_dict) for text in df[DESCRIPTION_COLUMN]]

#     print("🤖 Making predictions with the models...")
#     predicted_types = predict_with_hf_model(preprocessed_texts, models["type"], tokenizers["type"], device)
#     predicted_explanations = predict_with_hf_model(preprocessed_texts, models["explanation"], tokenizers["explanation"], device)

#     print("🗺️ Applying mappings...")
#     predicted_work_orders = [mappings["explanation_to_work_order"].get(exp, "Unknown") for exp in predicted_explanations]
#     predicted_categories = [mappings["work_order_to_category"].get(wo, "Unknown") for wo in predicted_work_orders]
#     predicted_work_priorities = [mappings["work_order_to_work_priority"].get(wo, "Unknown") for wo in predicted_work_orders]

#     # Add prediction columns - all original columns are preserved
#     df['predicted_type'] = predicted_types
#     df['predicted_explanation'] = predicted_explanations
#     df['predicted_work_order_category'] = predicted_work_orders
#     df['predicted_category'] = predicted_categories
#     df['predicted_work_priority'] = predicted_work_priorities
#     df['preprocessed_description'] = preprocessed_texts
    
#     # Ensure data types for key columns match your DDL
#     integer_columns = ['id', 'customer_id', 'have_phone_number', 'fixed_or_not', 'is_delete', 'version', 'history_flag', 'transfer_work_order']
#     for col in integer_columns:
#         if col in df.columns:
#             df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('Int64')
    
#     # Ensure datetime columns are properly formatted
#     datetime_columns = ['submit_time', 'created_time', 'updated_time', 'delete_time', 'migration_time']
#     for col in datetime_columns:
#         if col in df.columns:
#             df[col] = pd.to_datetime(df[col], errors='coerce')
    
#     print(f"✓ Added predictions to {len(df)} records")
#     print(f"  Original columns: {len(df.columns) - 6}")  # Subtract prediction columns
#     print(f"  Prediction columns: 6")
#     print(f"  Total columns: {len(df.columns)}")
    
#     return df

# # ================================
# # MAIN EXECUTION
# # ================================
# def main():
#     """Main execution function for MaxCompute/ODPS pipeline."""
#     print("🚀 Starting MaxCompute DataWorks ML Pipeline...")
    
#     # 1. Load models and assets
#     models, tokenizers, mappings, slang_dict, device = load_assets()
#     if not all([models, tokenizers, mappings]):
#         print("✗ Failed to load required assets. Exiting.")
#         return False
    
#     # 2. Connect to MaxCompute
#     odps = create_odps_connection()
#     if not odps:
#         print("✗ Failed to connect to MaxCompute. Exiting.")
#         return False
    
#     # 3. Fetch data from ODPS
#     if SOURCE_TABLE and not SOURCE_QUERY:
#         input_df = fetch_data_from_odps(odps, table_name=SOURCE_TABLE)
#     else:
#         input_df = fetch_data_from_odps(odps, query=SOURCE_QUERY)
        
#     if input_df is None or len(input_df) == 0:
#         print("✗ No data retrieved from MaxCompute. Exiting.")
#         return False
    
#     # 4. Run predictions
#     print(f"\n🔮 Running predictions on {len(input_df)} records...")
#     result_df = run_batch_predictions(input_df.copy(), models, tokenizers, mappings, slang_dict, device)
    
#     # 5. Save results back to MaxCompute
#     success = create_odps_output_table(odps, result_df, OUTPUT_TABLE_NAME, OUTPUT_PARTITION)
    
#     if success:
#         print(f"\n🎉 Pipeline completed successfully!")
#         print(f"   Input records: {len(input_df)}")
#         print(f"   Output table: {OUTPUT_TABLE_NAME}")
#         if OUTPUT_PARTITION:
#             print(f"   Partition: {OUTPUT_PARTITION}")
        
#         # Show sample predictions with original data
#         print(f"\n📊 Sample predictions with original data:")
#         sample_cols = ['id', 'customer_id', 'feedback_type', DESCRIPTION_COLUMN, 'predicted_type', 'predicted_explanation', 'predicted_work_priority']
#         available_cols = [col for col in sample_cols if col in result_df.columns]
#         print(result_df[available_cols].head(3).to_string(index=False, max_colwidth=50))
        
#     else:
#         print("✗ Pipeline failed during output creation.")
    
#     return success

# # ================================
# # DATAWORKS INTEGRATION HELPERS
# # ================================

# def dataworks_main():
#     """
#     Main function for DataWorks scheduling.
#     This function can be called from DataWorks PyODPS nodes.
#     """
#     import sys
    
#     try:
#         # Set up logging for DataWorks
#         import logging
#         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#         logger = logging.getLogger(__name__)
        
#         logger.info("Starting DataWorks ML Pipeline job...")
        
#         # Run the main pipeline
#         success = main()
        
#         if success:
#             logger.info("Pipeline completed successfully")
#             sys.exit(0)
#         else:
#             logger.error("Pipeline failed")
#             sys.exit(1)
            
#     except Exception as e:
#         print(f"✗ Fatal error in DataWorks pipeline: {e}")
#         import traceback
#         traceback.print_exc()
#         sys.exit(1)

# # ================================
# # EXECUTION
# # ================================
# if __name__ == "__main__":
#     main()

# # For DataWorks execution, uncomment the line below:
# dataworks_main()
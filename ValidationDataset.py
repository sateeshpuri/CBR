#!/usr/bin/env python
# coding: utf-8

# In[96]:


import sqlite3
import pandas as pd
import numpy as np

# Database file paths
ee_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/EE.db'
result_database2_db_path = '/Users/sateeshkumarpuri/Documents/Result_Database2.db'
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_ee_data(csj, database_path):
    conn = sqlite3.connect(database_path)
    query = "SELECT ItemCode, Quantity, UnitPrice FROM EE WHERE CSJ = ?"
    ee_data = pd.read_sql_query(query, conn, params=[csj])
    conn.close()
    return ee_data

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

# Function to calculate base weight and GSS
def calculate_base_weight_and_gss(user_df, db_df):
    project_amount = calculate_project_amount(user_df)
    user_df['UserWeight'] = user_df['Quantity'] * user_df['UnitPrice'] / project_amount

    # Calculate the TotalEstimate for each CSJ
    db_df['Product'] = db_df['TxDOTEE_Quantity'] * db_df['TxDOTEE_EngineerEstimate']
    total_estimates = db_df.groupby('CSJ')['Product'].sum().reset_index().rename(columns={'Product': 'TotalEstimate'})
    db_df = pd.merge(db_df, total_estimates, on='CSJ')

    # Calculate Base Weight using TotalEstimate
    db_df['BaseWeight'] = db_df['Product'] / db_df['TotalEstimate']

    gss_scores = {}
    for csj in db_df['CSJ'].unique():
        csj_df = db_df[db_df['CSJ'] == csj]
        gss = 0
        for _, user_row in user_df.iterrows():
            if user_row['ItemCode'] in csj_df['TxDOTEE_ItemCode'].values:
                item_df = csj_df[csj_df['TxDOTEE_ItemCode'] == user_row['ItemCode']]
                # Calculate the absolute difference between EE weight and New EE weight
                weight_difference = abs(user_row['UserWeight'] - item_df.iloc[0]['BaseWeight'])
                # Adjust the base weight by the difference
                adjusted_weight = item_df.iloc[0]['BaseWeight'] / (1 + weight_difference)
                # Add the adjusted weight to the GSS for this CSJ
                gss += adjusted_weight
        gss_scores[csj] = gss

    return gss_scores

def store_details_in_result_db(details, database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS Result (
                        CSJ TEXT,
                        ABI_FY2014_ItemCode TEXT,
                        ABI_FY2014_ItemDescription TEXT,
                        BidQty REAL,
                        NetCOQty REAL,
                        NetCOAmt REAL,
                        UNIQUE(CSJ, ABI_FY2014_ItemCode))''')  # UNIQUE constraint added

    # Insert data into the table, ignoring duplicates
    for detail in details:
        csj, item_code = detail[0], detail[1]
        # Check if the entry already exists
        cursor.execute('''SELECT * FROM Result WHERE CSJ = ? AND ABI_FY2014_ItemCode = ?''', (csj, item_code))
        if not cursor.fetchone():  # If no existing entry, insert the new data
            cursor.execute('''INSERT INTO Result (CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt)
                              VALUES (?, ?, ?, ?, ?, ?)''', detail)

    conn.commit()
    conn.close()


# Main execution
user_df = get_ee_data(csj_to_query, ee_db_path)
user_df['Quantity'] = pd.to_numeric(user_df['Quantity'], errors='coerce')
user_df['UnitPrice'] = pd.to_numeric(user_df['UnitPrice'], errors='coerce')
user_df = user_df.dropna(subset=['Quantity', 'UnitPrice'])


# Connect to the database and fetch required data
conn = sqlite3.connect(result_database2_db_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Store the top 10 CSJs with the highest GSS scores without printing
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
top_10_CSJs = top_gss_df['CSJ'].tolist()

# Fetch and store detailed items for missing and potentialCO in result.db
for detail_type in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_type, top_10_CSJs, result_database2_db_path)
    if detail_items:  # Check if there are any items to store
        store_details_in_result_db(detail_items, result_db_path)
        


# In[6]:


#modified # Do not Touch

import sqlite3
import pandas as pd
import numpy as np

# Database file paths
ee_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/EE.db'
result_database2_db_path = '/Users/sateeshkumarpuri/Documents/Result_Database2.db'
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'
# Assuming csj_to_query is a tuple of CSJs
csj_to_query = ('637714001','1004021')

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_ee_data(csj, database_path):
    conn = sqlite3.connect(database_path)
    query = "SELECT ItemCode, Quantity, UnitPrice FROM EE WHERE CSJ = ?"
    ee_data = pd.read_sql_query(query, conn, params=[csj])
    conn.close()
    return ee_data

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

# Function to calculate base weight and GSS
def calculate_base_weight_and_gss(user_df, db_df):
    project_amount = calculate_project_amount(user_df)
    user_df['UserWeight'] = user_df['Quantity'] * user_df['UnitPrice'] / project_amount

    # Calculate the TotalEstimate for each CSJ
    db_df['Product'] = db_df['TxDOTEE_Quantity'] * db_df['TxDOTEE_EngineerEstimate']
    total_estimates = db_df.groupby('CSJ')['Product'].sum().reset_index().rename(columns={'Product': 'TotalEstimate'})
    db_df = pd.merge(db_df, total_estimates, on='CSJ')

    # Calculate Base Weight using TotalEstimate
    db_df['BaseWeight'] = db_df['Product'] / db_df['TotalEstimate']

    gss_scores = {}
    for csj in db_df['CSJ'].unique():
        csj_df = db_df[db_df['CSJ'] == csj]
        gss = 0
        for _, user_row in user_df.iterrows():
            if user_row['ItemCode'] in csj_df['TxDOTEE_ItemCode'].values:
                item_df = csj_df[csj_df['TxDOTEE_ItemCode'] == user_row['ItemCode']]
                # Calculate the absolute difference between EE weight and New EE weight
                weight_difference = abs(user_row['UserWeight'] - item_df.iloc[0]['BaseWeight'])
                # Adjust the base weight by the difference
                adjusted_weight = item_df.iloc[0]['BaseWeight'] / (1 + weight_difference)
                # Add the adjusted weight to the GSS for this CSJ
                gss += adjusted_weight
        gss_scores[csj] = gss

    return gss_scores

def store_details_in_result_db(details, entered_csj, database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS Result (
                    CSJ TEXT,
                    ABI_FY2014_ItemCode TEXT,
                    ABI_FY2014_ItemDescription TEXT,
                    BidQty REAL,
                    NetCOQty REAL,
                    NetCOAmt REAL,
                    EnteredCSJ TEXT, 
                    UNIQUE(CSJ, ABI_FY2014_ItemCode))''')  # UNIQUE constraint added


    # Modify the INSERT INTO statement
    for detail in details:
        csj, item_code = detail[0], detail[1]
        cursor.execute('''SELECT * FROM Result WHERE CSJ = ? AND ABI_FY2014_ItemCode = ?''', (csj, item_code))
        if not cursor.fetchone():
            cursor.execute('''INSERT INTO Result (CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt, EnteredCSJ)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''', detail + (entered_csj,))

    conn.commit()
    conn.close()


# Main execution
for csj in csj_to_query:
    user_df = get_ee_data(csj, ee_db_path)
    user_df['Quantity'] = pd.to_numeric(user_df['Quantity'], errors='coerce')
    user_df['UnitPrice'] = pd.to_numeric(user_df['UnitPrice'], errors='coerce')
    user_df = user_df.dropna(subset=['Quantity', 'UnitPrice'])

    # Connect to the database and fetch required data
    conn = sqlite3.connect(result_database2_db_path)
    query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
    db_df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate GSS scores
    gss_scores = calculate_base_weight_and_gss(user_df, db_df)

    # Store the top 10 CSJs with the highest GSS scores without printing
    top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
    top_10_CSJs = top_gss_df['CSJ'].tolist()

    # Fetch and store detailed items for missing and potentialCO in result.db
    for detail_type in ['missing', 'potentialco']:
        detail_items = fetch_detail_items(detail_type, top_10_CSJs, result_database2_db_path)
        if detail_items:
            store_details_in_result_db(detail_items, csj, result_db_path)  # Pass the current CSJ for EnteredCSJ column

        


# In[143]:


#Modified and tested

import sqlite3
import pandas as pd
import numpy as np

# Database file paths
ee_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/EE.db'
result_database2_db_path = '/Users/sateeshkumarpuri/Documents/Result_Database2.db'
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'
# Assuming csj_to_query is a tuple of CSJs
csj_to_query = ('97402017','7002097','99702039','10006064','91000105','92039023','642265001','3504039',
                '638220001','638319001','14209044','53001007','636961001','631257001','633054001','16801029',
                '91138069','137802053','4508038','120601018','256101011', '235001057', '105101037', '632514001',
                '27511085', '90248732', '49502066', '90812031', '635100001', '636755001', '636327001', '637038001',
                '133002050', '180101051', '638290001', '638795001', '48601032', '3920008', '91847062', '7503021',
                '282501009', '31407069', '10201114', '3708041', '634295001', '634339001', '634364001', '90800104',
                '17305041', '26510032', '637386001', '637338001', '636930001', '106203051', '20203041', '20505048',
                '90400164', '133003003', '1303035', '629639001', '631223001', '630999001', '631017001', 
                '630946001', '4307119', '12801116', '56704022', '6903060', '75103041', '91273212', '120801028', 
                '90128094', '106101033', '638165001', '638289001', '635894001', '2701040', '21202039', '99703007', 
                '36702075', '320701011', '8806004', '9513040', '91600086', '710704001', '2807052', '351006015', 
                '10201116', '92802020', '33405032', '10902026', '63601038', '632857001', '402058', '16602044',
                '24804071', '4804101', '104803011', '4212085', '61003096', '17206102', '16809165', '91007077', 
                '634532001', '634853001', '632978001', '633201001', '636826001', '636823001', '636411001', 
                '306097', '638452001', '636535001', '638577001', '638785001', '91847128', '90328064', 
                '83303036', '12702043', '225002017', '4518038', '13307029', '1008056', '22005075', '24701057', 
                '4718086', '632007001', '632870001', '632867001', '632866001', '632877001', '632934001', 
                '631853001', '634239001', '633660001', '634410001', '633966001', '635628001', '638681001', 
                '38009092', '3601015', '4515012', '59802126', '36401157', '44002019', '7406242', '194702016', 
                '3504038', '7006041', '17103074')

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_ee_data(csj, database_path):
    conn = sqlite3.connect(database_path)
    query = "SELECT ItemCode, Quantity, UnitPrice FROM EE WHERE CSJ = ?"
    ee_data = pd.read_sql_query(query, conn, params=[csj])
    conn.close()
    return ee_data

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

# Function to calculate base weight and GSS
def calculate_base_weight_and_gss(user_df, db_df):
    project_amount = calculate_project_amount(user_df)
    user_df['UserWeight'] = user_df['Quantity'] * user_df['UnitPrice'] / project_amount

    # Calculate the TotalEstimate for each CSJ
    db_df['Product'] = db_df['TxDOTEE_Quantity'] * db_df['TxDOTEE_EngineerEstimate']
    total_estimates = db_df.groupby('CSJ')['Product'].sum().reset_index().rename(columns={'Product': 'TotalEstimate'})
    db_df = pd.merge(db_df, total_estimates, on='CSJ')

    # Calculate Base Weight using TotalEstimate
    db_df['BaseWeight'] = db_df['Product'] / db_df['TotalEstimate']

    gss_scores = {}
    for csj in db_df['CSJ'].unique():
        csj_df = db_df[db_df['CSJ'] == csj]
        gss = 0
        for _, user_row in user_df.iterrows():
            if user_row['ItemCode'] in csj_df['TxDOTEE_ItemCode'].values:
                item_df = csj_df[csj_df['TxDOTEE_ItemCode'] == user_row['ItemCode']]
                # Calculate the absolute difference between EE weight and New EE weight
                weight_difference = abs(user_row['UserWeight'] - item_df.iloc[0]['BaseWeight'])
                # Adjust the base weight by the difference
                adjusted_weight = item_df.iloc[0]['BaseWeight'] / (1 + weight_difference)
                # Add the adjusted weight to the GSS for this CSJ
                gss += adjusted_weight
        gss_scores[csj] = gss

    return gss_scores

def store_details_in_result_db(details, entered_csj, database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS Result (
                    CSJ TEXT,
                    ABI_FY2014_ItemCode TEXT,
                    ABI_FY2014_ItemDescription TEXT,
                    BidQty REAL,
                    NetCOQty REAL,
                    NetCOAmt REAL,
                    EnteredCSJ TEXT, 
                    UNIQUE(CSJ, ABI_FY2014_ItemCode))''')  # UNIQUE constraint added


    # Modify the INSERT INTO statement
    for detail in details:
        csj, item_code = detail[0], detail[1]
        cursor.execute('''SELECT * FROM Result WHERE CSJ = ? AND ABI_FY2014_ItemCode = ?''', (csj, item_code))
        if not cursor.fetchone():
            cursor.execute('''INSERT INTO Result (CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt, EnteredCSJ)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''', detail + (entered_csj,))

    conn.commit()
    conn.close()


# Main execution
for csj in csj_to_query:
    user_df = get_ee_data(csj, ee_db_path)
    user_df['Quantity'] = pd.to_numeric(user_df['Quantity'], errors='coerce')
    user_df['UnitPrice'] = pd.to_numeric(user_df['UnitPrice'], errors='coerce')
    user_df = user_df.dropna(subset=['Quantity', 'UnitPrice'])

    # Connect to the database and fetch required data
    conn = sqlite3.connect(result_database2_db_path)
    query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
    db_df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate GSS scores
    gss_scores = calculate_base_weight_and_gss(user_df, db_df)

    # Store the top 10 CSJs with the highest GSS scores without printing
    top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
    top_10_CSJs = top_gss_df['CSJ'].tolist()

    # Fetch and store detailed items for missing and potentialCO in result.db
    for detail_type in ['missing', 'potentialco']:
        detail_items = fetch_detail_items(detail_type, top_10_CSJs, result_database2_db_path)
        if detail_items:
            store_details_in_result_db(detail_items, csj, result_db_path)  # Pass the current CSJ for EnteredCSJ column

        


# In[161]:


import sqlite3
import pandas as pd

# Database paths
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'
validation_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Validation.db'

# Connect to databases
conn_result = sqlite3.connect(result_db_path)
conn_validation = sqlite3.connect(validation_db_path)

# Read data from databases
df_result = pd.read_sql_query("SELECT EnteredCSJ, ABI_FY2014_ItemCode FROM Result", conn_result)
df_validation = pd.read_sql_query("SELECT CSJ, ABI_FY2014_ItemCode FROM Validation WHERE NetCOQty != 0", conn_validation)

# Close database connections
conn_result.close()
conn_validation.close()

# Data cleaning: Convert Item Codes to integers and CSJs to strings
df_result['EnteredCSJ'] = df_result['EnteredCSJ'].astype(str).str.strip()
df_result['ABI_FY2014_ItemCode'] = df_result['ABI_FY2014_ItemCode'].astype(int)
df_validation['CSJ'] = df_validation['CSJ'].astype(str).str.strip()
df_validation['ABI_FY2014_ItemCode'] = df_validation['ABI_FY2014_ItemCode'].astype(int)

# Get unique EnteredCSJs from Result database
unique_enteredcsjs = set(df_result['EnteredCSJ'])
print("Number of unique EnteredCSJs:", len(unique_enteredcsjs))

# Initialize list to store accuracy for each EnteredCSJ
accuracies = []

for enteredcsj in unique_enteredcsjs:
    # Get unique Item Codes for the current EnteredCSJ
    result_item_codes = set(df_result[df_result['EnteredCSJ'] == enteredcsj]['ABI_FY2014_ItemCode'])
    validation_item_codes = set(df_validation[df_validation['CSJ'] == enteredcsj]['ABI_FY2014_ItemCode'])

    # Count matched Item Codes
    matched_count = sum(1 for code in result_item_codes if any(abs(code - v_code) <= 1000 for v_code in validation_item_codes))

    # Calculate accuracy and cap it at 1.0 if it exceeds
    total_validation_codes = len(validation_item_codes)
    if total_validation_codes > 0:
        accuracy = min(matched_count / total_validation_codes, 1.0)
        accuracies.append(accuracy)
        print(f"EnteredCSJ: {enteredcsj}, Matched: {matched_count}, Total: {total_validation_codes}, Accuracy: {accuracy}")

# Calculate mean accuracy across all EnteredCSJs
mean_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0

# Display the mean accuracy
print("Mean Accuracy: {:.2f}%".format(mean_accuracy * 100))


# In[166]:


import sqlite3
import pandas as pd

# Database paths
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'
validation_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Validation.db'

# Connect to databases
conn_result = sqlite3.connect(result_db_path)
conn_validation = sqlite3.connect(validation_db_path)

# Read data from databases
df_result = pd.read_sql_query("SELECT EnteredCSJ, ABI_FY2014_ItemCode FROM Result", conn_result)
df_validation = pd.read_sql_query("SELECT CSJ, ABI_FY2014_ItemCode FROM Validation WHERE NetCOQty != 0", conn_validation)

# Close database connections
conn_result.close()
conn_validation.close()

# Data cleaning: Convert Item Codes to integers and CSJs to strings
df_result['EnteredCSJ'] = df_result['EnteredCSJ'].astype(str).str.strip()
df_result['ABI_FY2014_ItemCode'] = df_result['ABI_FY2014_ItemCode'].astype(int)
df_validation['CSJ'] = df_validation['CSJ'].astype(str).str.strip()
df_validation['ABI_FY2014_ItemCode'] = df_validation['ABI_FY2014_ItemCode'].astype(int)

# Initialize variables for confusion matrix
TP = 0
FN = 0

# Process each EnteredCSJ
for enteredcsj in set(df_result['EnteredCSJ']):
    # Filter Item Codes for the current EnteredCSJ
    result_item_codes = set(df_result[df_result['EnteredCSJ'] == enteredcsj]['ABI_FY2014_ItemCode'])
    validation_item_codes = set(df_validation[df_validation['CSJ'] == enteredcsj]['ABI_FY2014_ItemCode'])

    # Count True Positives (TP) and False Negatives (FN)
    for v_code in validation_item_codes:
        if any(abs(v_code - code) <= 1000 for code in result_item_codes):
            TP += 1  # True Positive
        else:
            FN += 1  # False Negative

# Calculate accuracy
total = TP + FN
accuracy = TP / total if total > 0 else 0

# Display results
print("Confusion Matrix:")
print(f"True Positives (TP): {TP}")
print(f"False Negatives (FN): {FN}")
print(f"\nAccuracy: {accuracy:.2f}")


# In[167]:


import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

# Database paths
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'
validation_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Validation.db'

# Connect to databases
conn_result = sqlite3.connect(result_db_path)
conn_validation = sqlite3.connect(validation_db_path)

# Read data from databases
df_result = pd.read_sql_query("SELECT EnteredCSJ, ABI_FY2014_ItemCode FROM Result", conn_result)
df_validation = pd.read_sql_query("SELECT CSJ, ABI_FY2014_ItemCode FROM Validation WHERE NetCOQty != 0", conn_validation)

# Close database connections
conn_result.close()
conn_validation.close()

# Convert Item Codes to integers and CSJs to strings
df_result['EnteredCSJ'] = df_result['EnteredCSJ'].astype(str).str.strip()
df_result['ABI_FY2014_ItemCode'] = df_result['ABI_FY2014_ItemCode'].astype(int)
df_validation['CSJ'] = df_validation['CSJ'].astype(str).str.strip()
df_validation['ABI_FY2014_ItemCode'] = df_validation['ABI_FY2014_ItemCode'].astype(int)

# Initialize list to store accuracy for each EnteredCSJ
accuracies = []

for enteredcsj in set(df_result['EnteredCSJ']):
    # Get unique Item Codes for the current EnteredCSJ
    result_item_codes = set(df_result[df_result['EnteredCSJ'] == enteredcsj]['ABI_FY2014_ItemCode'])
    validation_item_codes = set(df_validation[df_validation['CSJ'] == enteredcsj]['ABI_FY2014_ItemCode'])

    # Count matched Item Codes
    matched_count = sum(1 for code in result_item_codes if any(abs(code - v_code) <= 1000 for v_code in validation_item_codes))

    # Calculate and cap accuracy
    total_validation_codes = len(validation_item_codes)
    if total_validation_codes > 0:
        accuracy = min(matched_count / total_validation_codes, 1.0)
        accuracies.append(accuracy)

# Categorize accuracies
categories = {'1': 0, '0.75 to 1': 0, '> 0.5': 0, '0.1 to 0.5': 0}
for acc in accuracies:
    if acc == 1:
        categories['1'] += 1
    elif 0.75 <= acc < 1:
        categories['0.75 to 1'] += 1
    elif 0.5 < acc < 0.75:
        categories['> 0.5'] += 1
    elif 0.1 <= acc <= 0.5:
        categories['0.1 to 0.5'] += 1

# Plotting
plt.bar(categories.keys(), categories.values())
plt.xlabel('Accuracy Range')
plt.ylabel('Number of Projects')
plt.title('Distribution of Project Accuracies')
plt.show()


# In[ ]:


#Modified and tested

import sqlite3
import pandas as pd
import numpy as np

# Database file paths
ee_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/EE.db'
result_database2_db_path = '/Users/sateeshkumarpuri/Documents/Result_Database2.db'
result_db_path = '/Users/sateeshkumarpuri/Documents/Validation Method/Result.db'
# Assuming csj_to_query is a tuple of CSJs
csj_to_query = ('97402017', '7002097', '99702039', '10006064', '91000105', '92039023', '642265001', '3504039',
'638220001', '638319001', '14209044', '53001007', '636961001', '631257001', '633054001', '16801029',
'91138069', '137802053', '4508038', '120601018', '256101011', '235001057', '105101037', '632514001',
'27511085', '90248732', '49502066', '90812031', '635100001', '636755001', '636327001', '637038001',
'133002050', '180101051', '638290001', '638795001', '48601032', '3920008', '91847062', '7503021',
'282501009', '31407069', '10201114', '3708041', '634295001', '634339001', '634364001', '90800104',
'17305041', '26510032', '637386001', '637338001', '636930001', '106203051', '20203041', '20505048',
'90400164', '133003003', '1303035', '629639001', '631223001', '630999001', '631017001', '630946001',
'4307119', '12801116', '56704022', '6903060', '75103041', '91273212', '120801028', '90128094',
'106101033', '638165001', '638289001', '635894001', '2701040', '21202039', '99703007', '36702075',
'320701011', '8806004', '9513040', '91600086', '710704001', '2807052', '351006015', '10201116',
'92802020', '33405032', '10902026', '63601038', '632857001', '402058', '16602044', '24804071',
'4804101', '104803011', '4212085', '61003096', '17206102', '16809165', '91007077', '634532001',
'634853001', '632978001', '633201001', '636826001', '636823001', '636411001', '306097', '638452001',
'636535001', '638577001', '638785001', '91847128', '90328064', '83303036', '12702043', '225002017',
'4518038', '13307029', '1008056', '22005075', '24701057', '4718086', '632007001', '632870001',
'632867001', '632866001', '632877001', '632934001', '631853001', '634239001', '633660001', '634410001',
'633966001', '635628001', '638681001', '38009092', '3601015', '4515012', '59802126', '36401157',
'44002019', '7406242', '194702016', '3504038', '7006041', '17103074', '16701117', '634684001',
'634000001', '635297001', '635953001', '636624001', '636623001', '315801044', '27117175', '92030077',
'72002085', '244601026', '32003097', '25107025', '255301117', '118802096', '639297001', '638820001',
'639558001', '631247001', '631314001', '631315001', '76403018', '17301050', '637397001', '638155001',
'639154001', '4107111', '6404043', '95708027', '5404107', '13308035', '638900001', '1605118',
'1011070', '237405082', '92322021', '7405097', '18301043', '20016019', '35706022', '72003141',
'90600192', '281301008', '92000124', '4604065', '632113001', '634399001', '633225001', '633263001',
'16007033', '141503012', '635990001', '636454001', '636626001', '636631001', '636451001', '636305001',
'312301014', '167302017', '13902025', '17603135', '142901033', '6310017', '117903010', '631353001',
'632890001', '632217001', '632892001', '636613001', '636628001', '148002036', '107701026', '203203013',
'90521004', '2713240', '638132001', '638207001', '638475001', '640724001', '708030', '69301036',
'11404069', '19202056', '407130', '38913062', '634623001', '635292001', '634784001', '5501029',
'37404035', '90119191', '153902034', '92038253', '637581001', '637523001', '637376001', '639206001',
'639621001', '17208053', '92200062', '90500090', '91319028', '90700166', '92102140', '55002038',
'5318042', '632572001', '632233001', '632579001', '630791001', '630448001', '16004051', '255202034',
'92317079', '91000113', '91400427', '185401048', '814133', '2602035', '8308053', '6207096',
'635949001', '636409001', '636415001', '638144001', '637714001', '638269001', '638141001', '602121',
'49506034', '73001011', '1019013', '96401009', '637276001', '635095001', '1110027', '69301037',
'4702154', '35406031', '4314026', '9802028', '637008001', '636970001', '637248001', '639054001',
'638370001', '638854001', '639072001', '638880001', '638027001', '4801064', '4301080', '91328064',
'631670001', '631742001', '632049001', '42803012', '90290178', '909103', '635304001', '636519001',
'639523001', '91400452', '300601007', '78001017', '18102032', '25804036', '11705057', '227001023',
'1708111', '91600217', '90238129', '1401025', '91034033', '636417001', '636572001', '636011001',
'638330001', '638157001', '637859001', '6405061', '106101032', '90517015', '44901019', '1013092',
'341703023', '69802052', '633436001', '633742001', '633343001', '632930001', '631938001', '2713212',
'5406104', '911243', '184201016', '43201065', '1902034', '633379001', '632047001', '633677001',
'632078001', '633869001', '636774001', '6807051', '32802043', '35304095', '90200148', '1509188',
'239901076', '52703024', '26601075', '26503048', '90500091', '11411076', '631123001', '631133001',
'631002001', '630962001', '631113001', '631083001', '630217001', '630839001', '629591001', '630449001',
'67501072', '54504051', '303401005', '50201214', '1416276', '26407032', '245202119', '101301034',
'91400446', '68303039', '23703035', '92306058', '73602017', '20308017', '24908044', '9214096',
'632818001', '633210001', '636074001', '635821001', '636108001', '636128001', '91035033', '11411080',
'90290113', '252402025', '90500122', '91012134', '633845001', '633746001', '634145001', '632648001',
'636895001', '636376001', '637140001', '637627001', '22901046', '91545055', '637723001', '635992001',
'637944001', '637945001', '6601021', '6404042', '11304027', '90200201', '17605197', '91400432',
'90290036', '69803099', '634374001', '634924001', '634794001', '636998001', '636528001', '636853001',
'636619001', '636928001', '636929001', '8805092', '21306044', '86702019', '6404040', '92000125',
'13805057', '634481001', '634544001', '634867001', '637737001', '637738001', '637745001', '28703032',
'91328066', '4912115', '6306099', '94402011', '90290088', '91500226', '5504031', '8101051',
'161502016', '47904050', '37902032', '140701025', '4906081', '25202060', '607085', '4502039',
'50003637', '50704045', '44002022', '90290076', '91035038', '20014097', '633896001', '634531001',
'631063001', '642604001', '642683001', '631154001', '6706052', '63204034', '633946001', '633568001',
'633669001', '633671001', '633792001', '633574001', '633866001', '632687001', '631813001', '214901010',
'18304053', '160601035', '11313177', '21110046', '78301098', '20004024', '281501008', '135701026',
'29110110', '36703037', '160101033', '40002053', '635103001', '635384001', '635081001', '633766001',
'634206001', '634207001', '633871001', '637362001', '636564001', '19503101', '90700198', '17702095',
'66802022', '638856001', '638147001', '2706046', '20002031', '49301013', '91727044', '1011069',
'33904035', '141401019', '606103', '267902008', '53502046', '12405029', '91237228', '106804119',
'20014085', '90715005', '245202118', '70003142', '631841001', '632821001', '632788001', '632495001',
'632962001', '632753001', '633199001', '6306098', '115701014', '83102020', '3917196', '90300088',
'19603276', '90239021', '17205122', '1603119', '40103026', '59802110', '15004047', '20014086',
'90315085', '91329054', '632607001', '632737001', '632631001', '633844001', '633847001', '632481001',
'633417001', '4308085', '237403074', '69401031', '23101057', '90833100', '91903062', '637700001',
'638022001', '638023001', '69702039', '213903021', '25504094', '30901043', '15001028', '102055',
'4306095', '4309139', '1305064', '91400426', '633153001', '633177001', '633265001', '631043001',
'632980001', '634954001', '634675001', '640486001', '630735001', '631667001', '631837001', '632107001',
'306094', '309702016', '632976001', '632748001', '633545001', '632983001', '633156001', '632933001',
'633335001', '633425001', '633705001', '635740001', '635785001', '635574001', '635558001', '169001132',
'255205002', '90922180', '176703020', '91426009', '637396001', '637580001', '637611001', '639609001',
'638675001', '639506001', '639780001', '639844001', '1603103', '13412005', '90200205', '8906090',
'25202054', '90938071', '4308075', '229601052', '2505021', '11701052', '180102017', '346502011',
'304801017', '14404043', '90200163', '2502223', '17105098', '17701112', '91404309', '633344001',
'634479001', '634596001', '634764001', '634171001', '634174001', '637425001', '639281001', '642612001',
'639302001', '638891001', '638809001', '637210001', '55002049', '90500107', '91847179', '202101013',
'54301065', '215002009', '635358001', '635192001', '6205062', '168505115', '57301040', '119701021',
'91511032', '639003001', '638888001', '638183001', '640094001', '191902037', '91329031', '72002091',
'5904046', '3306107', '1004021', '7406228', '19607031', '59802047', '903045', '631414001',
'631450001', '631659001', '53504030', '329201014', '319202009', '7905054', '4706159', '11810060',
'1511063', '26505079', '91629006', '7201054', '156701037', '47901019', '1013094', '81602079',
'2601027', '97802078', '2401103', '18202038', '45302018', '255203063', '631260001', '632742001',
'632489001', '631859001', '80903040', '90400195', '4801063', '8907155', '4808052', '607080',
'4105052', '39103031', '636340001', '636339001', '635596001', '636129001', '635374001', '632119001',
'631958001', '71501014', '2507065', '91400414', '222401100', '633422001', '633675001', '6002034',
'4909088', '190201033', '5904049', '13402060', '638122001', '640975001', '633253001', '632485001',
'632955001', '633407001', '74704076', '6505153', '13301052', '68302080', '18301047', '92200073',
'33403021', '91500229', '635515001', '634266001', '635622001', '635775001', '636900001', '634838001',
'32710057', '362101012', '90834022', '20601049', '91309065', '49508098', '38701014', '92030079',
'3401126', '19101063', '15302040', '223001013', '150302011', '212103160', '14501031', '315801032',
'501109', '201095', '90200213', '91273195', '66601027', '8303055', '14601051', '631895001',
'631918001', '631612001', '633114001', '632717001', '636431001', '638826001', '638829001', '636434001',
'636095001', '636097001', '635960001', '637297001', '637301001', '635035001', '639409001', '639493001',
'639311001', '12504033', '45703028', '10005185', '11303031', '34503023', '69101042', '633703001',
'633546001', '633891001', '633822001', '633903001', '636887001', '635087001', '637447001', '638193001',
'638195001', '638194001', '637949001', '638266001', '20401068', '90300116', '90512044', '57301039',
'635917001', '636000001', '634808001', '635912001', '636300001', '639442001', '639355001', '23116032',
'133001062', '13908037', '90600156', '20012015', '92102318', '25407009', '603132', '632995001',
'632557001', '632558001', '72203014', '17502093', '4509101', '91322046', '32003100', '633919001',
'635321001', '635183001', '635577001', '629911001', '630459001', '630731001', '630769001', '634771001',
'634772001', '634773001', '635018001', '634799001', '634818001', '91272353', '638360001', '637592001',
'636463001', '636609001', '636909001', '636910001', '9804036', '629944001', '1510063', '354502010',
'11104035', '115002006', '91317039', '92406603', '92407017', '54301066', '634269001', '635683001',
'635692001', '640211001', '640291001', '640293001', '640925001', '90290071', '4511038', '75204024',
'1602148', '286601009', '57304015', '90248576', '26207033', '631505001', '631959001', '631001001',
'637329001', '632469001', '632477001', '631698001', '631702001', '43702017', '27107322', '8907156',
'13806044', '49507076', '4308078', '15503037', '157504017', '635288001', '634973001', '635151001',
'635230001', '635270001', '633252001', '633867001', '633800001', '633801001', '636232001', '638085001',
'90250131', '173501017', '1423038', '91100082', '2002019', '90819026', '88302087', '91016151',
'634038001', '633557001', '636278001', '635767001', '639685001', '639752001', '637704001', '631395001',
'631491001', '631482001', '631411001', '639065001', '639229001', '638941001', '636032001', '636753001',
'632850001', '632762001', '638526001', '638684001', '638571001', '638522001', '641439001', '641457001',
'639592001', '643059001', '632622001', '635130001', '635310001', '638048001', '903047', '259001020',
'118601098', '8918003', '92000121', '26502035', '632809001', '634257001', '634272001', '634438001',
'634169001', '635594001', '635742001', '635489001', '636429001', '636430001', '637310001', '637311001',
'632117001', '632019001', '633262001', '635791001', '631029001', '638385001', '636245001', '636575001',
'636239001', '636170001', '636437001', '636471001', '635177001', '639504001', '633691001', '633697001',
'633689001', '633365001', '633886001', '631996001', '633961001', '631991001', '631543001', '633555001',
'642029001', '642346001', '13407070', '160101031', '167201004', '80801058', '50801360', '15005047',
'75103037', '91847118', '11101092', '46304026', '210001054', '319901008', '11412011', '32005015',
'5001080', '3916066', '19401014', '91404229', '265401022', '39201073', '109103012', '106801219',
'12001019', '25905077', '631884001', '631738001', '630980001', '631561001', '629000001', '632828001',
'632992001', '637512001', '637515001', '635379001', '640233001', '638978001', '642173001', '639082001',
'639084001', '641485001', '642017001', '631869001', '633395001', '633494001', '631285001', '638963001',
'636521001', '633977001', '633242001', '633439001', '633716001', '634076001', '633306001', '633309001',
'637491001', '637402001', '636478001', '635604001', '640993001', '60501060', '212102150', '135204014',
'61006088', '91400396', '91139038', '26505081', '2806078', '78302096', '92038256', '91400406',
'205045', '631753001', '631529001', '637255001', '634346001', '634963001', '634964001', '634968001',
'638841001', '638517001', '642037001', '638961001', '641483001', '631488001', '633071001', '628761001',
'637824001', '637821001', '637906001', '637907001', '637908001', '637741001', '637928001', '637930001',
'634757001', '634782001', '633938001', '634492001', '634497001', '634538001', '636667001', '641728001',
'642600001', '634051001', '637370001', '637166001', '639141001', '631677001', '638090001', '638025001',
'638398001', '641023001', '641119001', '635278001', '635682001', '638583001', '638601001', '636939001',
'636513001', '638359001', '636700001', '638959001', '636083001', '636085001', '636087001', '640035001',
'633310001', '635415001', '633410001', '639034001', '638600001', '639705001', '640681001', '641730001',
'632320001', '634358001', '635411001', '640879001', '631974001', '631323001', '632659001', '632097001',
'632345001', '631598001', '636874001', '636674001', '630847001', '637645001', '637648001', '635671001',
'637839001', '637842001', '637838001', '634948001', '631638001', '631768001', '631641001', '639739001',
'632539001', '632674001', '639468001', '639632001', '635338001', '637981001', '638102001', '638204001',
'637864001', '639934001', '640364001', '638771001', '637868001', '637887001', '638407001', '639453001',
'636330001', '637609001', '634612001', '631079001', '642242001', '635106001', '634687001', '636182001',
'635971001', '635973001', '635328001', '635842001', '633292001', '641463001', '634542001', '638552001',
'637320001', '639349001', '635322001', '638297001', '638304001', '637726001', '637773001', '639894001',
'631350001', '631520001', '631817001', '631820001', '639262001', '639234001', '639235001', '639236001',
'642247001', '642289001', '630906001', '630909001', '631093001', '630911001', '631077001', '633813001',
'635099001', '633788001', '637198001', '635951001', '636670001', '632564001', '632917001', '640096001',
'641017001', '638232001', '638234001', '638759001', '638620001', '638750001', '636907001', '637012001',
'637428001', '635162001', '633354001', '632994001', '633003001', '638538001', '637754001', '637605001',
'635236001', '637528001', '638527001', '636855001', '637331001', '637332001', '633594001', '632215001',
'636571001', '636602001', '636827001', '636488001', '636577001', '631540001', '636580001', '631891001',
'632359001', '631301001', '631171001', '631838001', '631646001', '631527001', '640040001', '639028001',
'637359001', '636594001', '636598001', '639313001', '641752001', '633673001', '633498001', '632064001',
'632637001', '640036001', '639886001', '639907001', '639908001', '637749001', '641215001', '635707001',
'640124001', '636764001', '636829001', '635204001', '635246001', '638917001', '635870001', '640056001',
'640066001', '637715001', '632724001', '636860001', '631896001', '638224001', '638228001', '632492001',
'639104001', '639107001', '630509001', '630918001', '637507001', '635146001', '640821001', '640655001',
'640675001', '634086001', '631231001', '635674001', '635330001', '633146001', '639366001', '638278001',
'643174001', '635381001', '634908001', '638101001', '638375001', '632609001', '632891001', '638174001',
'639171001', '639177001', '636942001', '636146001', '635786001', '635598001', '640482001', '639513001',
'632081001', '635455001', '637650001', '637652001', '637778001', '635726001', '639389001', '635128001',
'637184001', '637185001', '637103001', '635780001', '638742001', '635181001', '637761001', '637762001',
'639732001', '639472001', '641889001', '637132001', '633080001', '633088001', '632644001', '632973001',
'636576001', '640229001', '631925001', '635343001', '635617001', '635632001', '635655001', '635659001',
'635662001', '637526001', '637854001', '637856001', '632311001', '641722001', '640156001', '640077001',
'641427001', '641123001', '633372001', '635695001', '634140001', '639140001', '638635001', '637955001',
'631588001', '635613001', '636091001', '632009001', '637259001', '637272001', '633158001', '631876001',
'632127001', '638515001', '638419001', '638629001', '630749001', '630205001', '638839001', '635869001',
'635820001', '637828001', '637282001', '637623001', '13506034', '640753001', '640834001', '639256001',
'640273001', '638281001', '631678001', '640412001', '636836001', '632161001', '639859001', '639863001',
'635800001', '640277001', '633810001')
            

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_ee_data(csj, database_path):
    conn = sqlite3.connect(database_path)
    query = "SELECT ItemCode, Quantity, UnitPrice FROM EE WHERE CSJ = ?"
    ee_data = pd.read_sql_query(query, conn, params=[csj])
    conn.close()
    return ee_data

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

# Function to calculate base weight and GSS
def calculate_base_weight_and_gss(user_df, db_df):
    project_amount = calculate_project_amount(user_df)
    user_df['UserWeight'] = user_df['Quantity'] * user_df['UnitPrice'] / project_amount

    # Calculate the TotalEstimate for each CSJ
    db_df['Product'] = db_df['TxDOTEE_Quantity'] * db_df['TxDOTEE_EngineerEstimate']
    total_estimates = db_df.groupby('CSJ')['Product'].sum().reset_index().rename(columns={'Product': 'TotalEstimate'})
    db_df = pd.merge(db_df, total_estimates, on='CSJ')

    # Calculate Base Weight using TotalEstimate
    db_df['BaseWeight'] = db_df['Product'] / db_df['TotalEstimate']

    gss_scores = {}
    for csj in db_df['CSJ'].unique():
        csj_df = db_df[db_df['CSJ'] == csj]
        gss = 0
        for _, user_row in user_df.iterrows():
            if user_row['ItemCode'] in csj_df['TxDOTEE_ItemCode'].values:
                item_df = csj_df[csj_df['TxDOTEE_ItemCode'] == user_row['ItemCode']]
                # Calculate the absolute difference between EE weight and New EE weight
                weight_difference = abs(user_row['UserWeight'] - item_df.iloc[0]['BaseWeight'])
                # Adjust the base weight by the difference
                adjusted_weight = item_df.iloc[0]['BaseWeight'] / (1 + weight_difference)
                # Add the adjusted weight to the GSS for this CSJ
                gss += adjusted_weight
        gss_scores[csj] = gss

    return gss_scores

def store_details_in_result_db(details, entered_csj, database_path):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute('''CREATE TABLE IF NOT EXISTS Result (
                    CSJ TEXT,
                    ABI_FY2014_ItemCode TEXT,
                    ABI_FY2014_ItemDescription TEXT,
                    BidQty REAL,
                    NetCOQty REAL,
                    NetCOAmt REAL,
                    EnteredCSJ TEXT, 
                    UNIQUE(CSJ, ABI_FY2014_ItemCode))''')  # UNIQUE constraint added


    # Modify the INSERT INTO statement
    for detail in details:
        csj, item_code = detail[0], detail[1]
        cursor.execute('''SELECT * FROM Result WHERE CSJ = ? AND ABI_FY2014_ItemCode = ?''', (csj, item_code))
        if not cursor.fetchone():
            cursor.execute('''INSERT INTO Result (CSJ, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt, EnteredCSJ)
                              VALUES (?, ?, ?, ?, ?, ?, ?)''', detail + (entered_csj,))

    conn.commit()
    conn.close()


# Main execution
for csj in csj_to_query:
    user_df = get_ee_data(csj, ee_db_path)
    user_df['Quantity'] = pd.to_numeric(user_df['Quantity'], errors='coerce')
    user_df['UnitPrice'] = pd.to_numeric(user_df['UnitPrice'], errors='coerce')
    user_df = user_df.dropna(subset=['Quantity', 'UnitPrice'])

    # Connect to the database and fetch required data
    conn = sqlite3.connect(result_database2_db_path)
    query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
    db_df = pd.read_sql_query(query, conn)
    conn.close()

    # Calculate GSS scores
    gss_scores = calculate_base_weight_and_gss(user_df, db_df)

    # Store the top 10 CSJs with the highest GSS scores without printing
    top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
    top_10_CSJs = top_gss_df['CSJ'].tolist()

    # Fetch and store detailed items for missing and potentialCO in result.db
    for detail_type in ['missing', 'potentialco']:
        detail_items = fetch_detail_items(detail_type, top_10_CSJs, result_database2_db_path)
        if detail_items:
            store_details_in_result_db(detail_items, csj, result_db_path)  # Pass the current CSJ for EnteredCSJ column

        


# In[ ]:





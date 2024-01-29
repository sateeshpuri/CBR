#!/usr/bin/env python
# coding: utf-8

# In[1]:


#637714001

import sqlite3
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_user_input():
    user_data = []
    print("Enter the Work Item as Item Code, Quantity, and Unit Price. Type 'done' to finish.")
    while True:
        user_input = input("Enter Item Code, Quantity, Unit Price (comma-separated): ")
        if user_input.lower() == 'done':
            break
        item_code, quantity, unit_price = user_input.split(',')
        user_data.append({
            'ItemCode': item_code.strip(),
            'Quantity': float(quantity.strip()),
            'UnitPrice': float(unit_price.strip())
        })
    return pd.DataFrame(user_data)

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

# Define function to fetch and display project details
def fetch_project_details(csjs, database_path):
    conn = sqlite3.connect(database_path)
    project_details = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute("""SELECT CSJ, ProjectDescription, ProjectType, District, CESTAMT
                          FROM ResultTable
                          WHERE CSJ = ?""", (csj,))
        project_details.append(cursor.fetchone())
        cursor.close()
    conn.close()
    return project_details

# Define function to fetch and display detailed item information
def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

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


# Main execution

# Get user input for work items
user_df = get_user_input()

# Connect to the database and fetch required data
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(database_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Display the top 10 CSJs with the highest GSS scores
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
print("\nTop 10 CSJs with highest scores:")
print(top_gss_df.to_string(index=False))

# Fetch and display project details for the top 10 CSJs
top_10_CSJs = top_gss_df['CSJ'].tolist()
project_details = fetch_project_details(top_10_CSJs, database_path)
df_project_details = pd.DataFrame(project_details, columns=["CSJ", "ProjectDescription", "ProjectType", "District", "CESTAMT"])
print("\nProject details for the top 10 CSJs:")
print(df_project_details.to_string(index=False))

# Ask user to see detailed items for missing or potentialCO
detail_choice = input("\nDo you want to see details for 'missing' items or 'potentialCO'? Enter your choice: ").lower()
if detail_choice in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_choice, top_10_CSJs, database_path)
    df_detail_items = pd.DataFrame(detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
    print(f"\n{detail_choice.capitalize()} item details for the top 10 CSJs:")
    print(df_detail_items.to_string(index=False))

    # After showing the chosen detail, ask if the user wants to see the other detail type
    other_choice = 'potentialco' if detail_choice == 'missing' else 'missing'
    if input(f"\nPress Enter to see the {other_choice.capitalize()} Items or any other key to exit: ") == '':
        other_detail_items = fetch_detail_items(other_choice, top_10_CSJs, database_path)
        df_other_detail_items = pd.DataFrame(other_detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
        print(f"\n{other_choice.capitalize()} item details for the top 10 CSJs:")
        print(df_other_detail_items.to_string(index=False))
else:
    print("Invalid choice or details not requested.")


# In[2]:


#1004021

import sqlite3
import pandas as pd
import numpy as np

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_user_input():
    user_data = []
    print("Enter the Work Item as Item Code, Quantity, and Unit Price. Type 'done' to finish.")
    while True:
        user_input = input("Enter Item Code, Quantity, Unit Price (comma-separated): ")
        if user_input.lower() == 'done':
            break
        item_code, quantity, unit_price = user_input.split(',')
        user_data.append({
            'ItemCode': item_code.strip(),
            'Quantity': float(quantity.strip()),
            'UnitPrice': float(unit_price.strip())
        })
    return pd.DataFrame(user_data)

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

# Define function to fetch and display project details
def fetch_project_details(csjs, database_path):
    conn = sqlite3.connect(database_path)
    project_details = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute("""SELECT CSJ, ProjectDescription, ProjectType, District, CESTAMT
                          FROM ResultTable
                          WHERE CSJ = ?""", (csj,))
        project_details.append(cursor.fetchone())
        cursor.close()
    conn.close()
    return project_details

# Define function to fetch and display detailed item information
def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

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


# Main execution

# Get user input for work items
user_df = get_user_input()

# Connect to the database and fetch required data
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(database_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Display the top 10 CSJs with the highest GSS scores
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
print("\nTop 10 CSJs with highest scores:")
print(top_gss_df.to_string(index=False))

# Fetch and display project details for the top 10 CSJs
top_10_CSJs = top_gss_df['CSJ'].tolist()
project_details = fetch_project_details(top_10_CSJs, database_path)
df_project_details = pd.DataFrame(project_details, columns=["CSJ", "ProjectDescription", "ProjectType", "District", "CESTAMT"])
print("\nProject details for the top 10 CSJs:")
print(df_project_details.to_string(index=False))

# Ask user to see detailed items for missing or potentialCO
detail_choice = input("\nDo you want to see details for 'missing' items or 'potentialCO'? Enter your choice: ").lower()
if detail_choice in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_choice, top_10_CSJs, database_path)
    df_detail_items = pd.DataFrame(detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
    print(f"\n{detail_choice.capitalize()} item details for the top 10 CSJs:")
    print(df_detail_items.to_string(index=False))

    # After showing the chosen detail, ask if the user wants to see the other detail type
    other_choice = 'potentialco' if detail_choice == 'missing' else 'missing'
    if input(f"\nPress Enter to see the {other_choice.capitalize()} Items or any other key to exit: ") == '':
        other_detail_items = fetch_detail_items(other_choice, top_10_CSJs, database_path)
        df_other_detail_items = pd.DataFrame(other_detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
        print(f"\n{other_choice.capitalize()} item details for the top 10 CSJs:")
        print(df_other_detail_items.to_string(index=False))
else:
    print("Invalid choice or details not requested.")


# In[4]:


#631314001

import sqlite3
import pandas as pd
import numpy as np

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_user_input():
    user_data = []
    print("Enter the Work Item as Item Code, Quantity, and Unit Price. Type 'done' to finish.")
    while True:
        user_input = input("Enter Item Code, Quantity, Unit Price (comma-separated): ")
        if user_input.lower() == 'done':
            break
        item_code, quantity, unit_price = user_input.split(',')
        user_data.append({
            'ItemCode': item_code.strip(),
            'Quantity': float(quantity.strip()),
            'UnitPrice': float(unit_price.strip())
        })
    return pd.DataFrame(user_data)

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

# Define function to fetch and display project details
def fetch_project_details(csjs, database_path):
    conn = sqlite3.connect(database_path)
    project_details = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute("""SELECT CSJ, ProjectDescription, ProjectType, District, CESTAMT
                          FROM ResultTable
                          WHERE CSJ = ?""", (csj,))
        project_details.append(cursor.fetchone())
        cursor.close()
    conn.close()
    return project_details

# Define function to fetch and display detailed item information
def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

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


# Main execution

# Get user input for work items
user_df = get_user_input()

# Connect to the database and fetch required data
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(database_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Display the top 10 CSJs with the highest GSS scores
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
print("\nTop 10 CSJs with highest scores:")
print(top_gss_df.to_string(index=False))

# Fetch and display project details for the top 10 CSJs
top_10_CSJs = top_gss_df['CSJ'].tolist()
project_details = fetch_project_details(top_10_CSJs, database_path)
df_project_details = pd.DataFrame(project_details, columns=["CSJ", "ProjectDescription", "ProjectType", "District", "CESTAMT"])
print("\nProject details for the top 10 CSJs:")
print(df_project_details.to_string(index=False))

# Ask user to see detailed items for missing or potentialCO
detail_choice = input("\nDo you want to see details for 'missing' items or 'potentialCO'? Enter your choice: ").lower()
if detail_choice in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_choice, top_10_CSJs, database_path)
    df_detail_items = pd.DataFrame(detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
    print(f"\n{detail_choice.capitalize()} item details for the top 10 CSJs:")
    print(df_detail_items.to_string(index=False))

    # After showing the chosen detail, ask if the user wants to see the other detail type
    other_choice = 'potentialco' if detail_choice == 'missing' else 'missing'
    if input(f"\nPress Enter to see the {other_choice.capitalize()} Items or any other key to exit: ") == '':
        other_detail_items = fetch_detail_items(other_choice, top_10_CSJs, database_path)
        df_other_detail_items = pd.DataFrame(other_detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
        print(f"\n{other_choice.capitalize()} item details for the top 10 CSJs:")
        print(df_other_detail_items.to_string(index=False))
else:
    print("Invalid choice or details not requested.")


# In[5]:


#91329031

import sqlite3
import pandas as pd
import numpy as np

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_user_input():
    user_data = []
    print("Enter the Work Item as Item Code, Quantity, and Unit Price. Type 'done' to finish.")
    while True:
        user_input = input("Enter Item Code, Quantity, Unit Price (comma-separated): ")
        if user_input.lower() == 'done':
            break
        item_code, quantity, unit_price = user_input.split(',')
        user_data.append({
            'ItemCode': item_code.strip(),
            'Quantity': float(quantity.strip()),
            'UnitPrice': float(unit_price.strip())
        })
    return pd.DataFrame(user_data)

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

# Define function to fetch and display project details
def fetch_project_details(csjs, database_path):
    conn = sqlite3.connect(database_path)
    project_details = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute("""SELECT CSJ, ProjectDescription, ProjectType, District, CESTAMT
                          FROM ResultTable
                          WHERE CSJ = ?""", (csj,))
        project_details.append(cursor.fetchone())
        cursor.close()
    conn.close()
    return project_details

# Define function to fetch and display detailed item information
def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

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


# Main execution

# Get user input for work items
user_df = get_user_input()

# Connect to the database and fetch required data
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(database_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Display the top 10 CSJs with the highest GSS scores
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
print("\nTop 10 CSJs with highest scores:")
print(top_gss_df.to_string(index=False))

# Fetch and display project details for the top 10 CSJs
top_10_CSJs = top_gss_df['CSJ'].tolist()
project_details = fetch_project_details(top_10_CSJs, database_path)
df_project_details = pd.DataFrame(project_details, columns=["CSJ", "ProjectDescription", "ProjectType", "District", "CESTAMT"])
print("\nProject details for the top 10 CSJs:")
print(df_project_details.to_string(index=False))

# Ask user to see detailed items for missing or potentialCO
detail_choice = input("\nDo you want to see details for 'missing' items or 'potentialCO'? Enter your choice: ").lower()
if detail_choice in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_choice, top_10_CSJs, database_path)
    df_detail_items = pd.DataFrame(detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
    print(f"\n{detail_choice.capitalize()} item details for the top 10 CSJs:")
    print(df_detail_items.to_string(index=False))

    # After showing the chosen detail, ask if the user wants to see the other detail type
    other_choice = 'potentialco' if detail_choice == 'missing' else 'missing'
    if input(f"\nPress Enter to see the {other_choice.capitalize()} Items or any other key to exit: ") == '':
        other_detail_items = fetch_detail_items(other_choice, top_10_CSJs, database_path)
        df_other_detail_items = pd.DataFrame(other_detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
        print(f"\n{other_choice.capitalize()} item details for the top 10 CSJs:")
        print(df_other_detail_items.to_string(index=False))
else:
    print("Invalid choice or details not requested.")


# In[6]:


#2507065

import sqlite3
import pandas as pd
import numpy as np

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_user_input():
    user_data = []
    print("Enter the Work Item as Item Code, Quantity, and Unit Price. Type 'done' to finish.")
    while True:
        user_input = input("Enter Item Code, Quantity, Unit Price (comma-separated): ")
        if user_input.lower() == 'done':
            break
        item_code, quantity, unit_price = user_input.split(',')
        user_data.append({
            'ItemCode': item_code.strip(),
            'Quantity': float(quantity.strip()),
            'UnitPrice': float(unit_price.strip())
        })
    return pd.DataFrame(user_data)

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

# Define function to fetch and display project details
def fetch_project_details(csjs, database_path):
    conn = sqlite3.connect(database_path)
    project_details = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute("""SELECT CSJ, ProjectDescription, ProjectType, District, CESTAMT
                          FROM ResultTable
                          WHERE CSJ = ?""", (csj,))
        project_details.append(cursor.fetchone())
        cursor.close()
    conn.close()
    return project_details

# Define function to fetch and display detailed item information
def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

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


# Main execution

# Get user input for work items
user_df = get_user_input()

# Connect to the database and fetch required data
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(database_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Display the top 10 CSJs with the highest GSS scores
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
print("\nTop 10 CSJs with highest scores:")
print(top_gss_df.to_string(index=False))

# Fetch and display project details for the top 10 CSJs
top_10_CSJs = top_gss_df['CSJ'].tolist()
project_details = fetch_project_details(top_10_CSJs, database_path)
df_project_details = pd.DataFrame(project_details, columns=["CSJ", "ProjectDescription", "ProjectType", "District", "CESTAMT"])
print("\nProject details for the top 10 CSJs:")
print(df_project_details.to_string(index=False))

# Ask user to see detailed items for missing or potentialCO
detail_choice = input("\nDo you want to see details for 'missing' items or 'potentialCO'? Enter your choice: ").lower()
if detail_choice in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_choice, top_10_CSJs, database_path)
    df_detail_items = pd.DataFrame(detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
    print(f"\n{detail_choice.capitalize()} item details for the top 10 CSJs:")
    print(df_detail_items.to_string(index=False))

    # After showing the chosen detail, ask if the user wants to see the other detail type
    other_choice = 'potentialco' if detail_choice == 'missing' else 'missing'
    if input(f"\nPress Enter to see the {other_choice.capitalize()} Items or any other key to exit: ") == '':
        other_detail_items = fetch_detail_items(other_choice, top_10_CSJs, database_path)
        df_other_detail_items = pd.DataFrame(other_detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
        print(f"\n{other_choice.capitalize()} item details for the top 10 CSJs:")
        print(df_other_detail_items.to_string(index=False))
else:
    print("Invalid choice or details not requested.")


# In[1]:


#91329031

import sqlite3
import pandas as pd
import numpy as np

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)

# Function definitions
def get_user_input():
    user_data = []
    print("Enter the Work Item as Item Code, Quantity, and Unit Price. Type 'done' to finish.")
    while True:
        user_input = input("Enter Item Code, Quantity, Unit Price (comma-separated): ")
        if user_input.lower() == 'done':
            break
        item_code, quantity, unit_price = user_input.split(',')
        user_data.append({
            'ItemCode': item_code.strip(),
            'Quantity': float(quantity.strip()),
            'UnitPrice': float(unit_price.strip())
        })
    return pd.DataFrame(user_data)

def calculate_project_amount(user_df):
    user_df['UserAmount'] = user_df['Quantity'] * user_df['UnitPrice']
    return user_df['UserAmount'].sum()

# Define function to fetch and display project details
def fetch_project_details(csjs, database_path):
    conn = sqlite3.connect(database_path)
    project_details = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute("""SELECT CSJ, ProjectDescription, ProjectType, District, CESTAMT
                          FROM ResultTable
                          WHERE CSJ = ?""", (csj,))
        project_details.append(cursor.fetchone())
        cursor.close()
    conn.close()
    return project_details

# Define function to fetch and display detailed item information
def fetch_detail_items(detail_choice, csjs, database_path):
    action_column = 'MismatchItem' if detail_choice == 'missing' else 'PotentialCO'
    conn = sqlite3.connect(database_path)
    detail_items = []
    for csj in csjs:
        cursor = conn.cursor()
        cursor.execute(f"""SELECT ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, BidQty, NetCOQty, NetCOAmt
                           FROM ResultTable
                           WHERE CSJ = ? AND {action_column} = 'Yes'""", (csj,))
        detail_items.extend(cursor.fetchall())
        cursor.close()
    conn.close()
    return detail_items

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


# Main execution

# Get user input for work items
user_df = get_user_input()

# Connect to the database and fetch required data
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(database_path)
query = "SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable"
db_df = pd.read_sql_query(query, conn)
conn.close()

# Calculate GSS scores
gss_scores = calculate_base_weight_and_gss(user_df, db_df)

# Display the top 10 CSJs with the highest GSS scores
top_gss_df = pd.DataFrame(list(gss_scores.items()), columns=['CSJ', 'Score']).sort_values(by='Score', ascending=False).head(10)
print("\nTop 10 CSJs with highest scores:")
print(top_gss_df.to_string(index=False))

# Fetch and display project details for the top 10 CSJs
top_10_CSJs = top_gss_df['CSJ'].tolist()
project_details = fetch_project_details(top_10_CSJs, database_path)
df_project_details = pd.DataFrame(project_details, columns=["CSJ", "ProjectDescription", "ProjectType", "District", "CESTAMT"])
print("\nProject details for the top 10 CSJs:")
print(df_project_details.to_string(index=False))

# Ask user to see detailed items for missing or potentialCO
detail_choice = input("\nDo you want to see details for 'missing' items or 'potentialCO'? Enter your choice: ").lower()
if detail_choice in ['missing', 'potentialco']:
    detail_items = fetch_detail_items(detail_choice, top_10_CSJs, database_path)
    df_detail_items = pd.DataFrame(detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
    print(f"\n{detail_choice.capitalize()} item details for the top 10 CSJs:")
    print(df_detail_items.to_string(index=False))

    # After showing the chosen detail, ask if the user wants to see the other detail type
    other_choice = 'potentialco' if detail_choice == 'missing' else 'missing'
    if input(f"\nPress Enter to see the {other_choice.capitalize()} Items or any other key to exit: ") == '':
        other_detail_items = fetch_detail_items(other_choice, top_10_CSJs, database_path)
        df_other_detail_items = pd.DataFrame(other_detail_items, columns=["ABI_FY2014_ItemCode", "ABI_FY2014_ItemDescription", "BidQty", "NetCOQty", "NetCOAmt"])
        print(f"\n{other_choice.capitalize()} item details for the top 10 CSJs:")
        print(df_other_detail_items.to_string(index=False))
else:
    print("Invalid choice or details not requested.")


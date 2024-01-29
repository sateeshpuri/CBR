#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sqlite3

# Connect to the database
db_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Ask the user for their project amount
user_amount = float(input("Please enter your project amount: "))

# Extract amounts and corresponding CSJs from the database
cursor.execute("SELECT DISTINCT CESTAMT, CSJ FROM ResultTable")
rows = cursor.fetchall()

# If there's no data, exit
if not rows:
    print("No data found in the database.")
    exit()

# Calculate the squared error for each distinct row
errors = [(user_amount - float(row[0]))**2 for row in rows]

# Convert errors into scores (inverse of error)
max_error = max(errors)  # to avoid division by zero and normalize scores
scores = [(max_error / (error + 1e-10), row[1]) for error, row in zip(errors, rows)]

# Normalize the scores
max_score = max([score[0] for score in scores])
normalized_scores = [(score[0] / max_score, score[1]) for score in scores]

# Sort the scores and get the top 10
top_10_scores = sorted(normalized_scores, key=lambda x: x[0], reverse=True)[:10]

# Present the table
print("CSJ\t\tNormalized Score")
print("-----------------------------")
for score, csj in top_10_scores:
    print(f"{csj}\t\t{score:.10f}")

# Close the database connection
conn.close()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Decision Tree Attribute Weightage

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
import sqlite3
from imblearn.over_sampling import SMOTE

# Connect to the SQLite database and load the DataFrame
conn = sqlite3.connect('/Users/sateeshkumarpuri/Documents/Result_Database2.db')
query = "SELECT * FROM ResultTable"
df = pd.read_sql(query, conn)
conn.close()

# Selecting specific columns for training
X = df[['ProjectDescription', 'ProjectNumber', 
        'City', 'CESTAMT', 
        'ABI_FY2014_ItemCode']]
Y = df['PotentialCO']

# Encode categorical variables
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)

# Applying SMOTE
sm = SMOTE(random_state=25)
X_res, Y_res = sm.fit_resample(X, Y)

# Then do your train-test split on the resampled data
X_train, X_test, y_train, y_test = train_test_split(X_res, Y_res, test_size=0.35, random_state=1234567)


# Create Random Forest Classifier
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=10000, class_weight={'Yes': 1.5, 'No': 1.0})

# Train the Classifier
clf = clf.fit(X_train, y_train)

# Get predicted probabilities
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Set custom threshold
threshold = 0.5
y_pred_custom_threshold = ['Yes' if prob > threshold else 'No' for prob in y_pred_proba]

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred_custom_threshold)
print("Confusion Matrix with Custom Threshold:\n", conf_matrix)

# Compute recall from the confusion matrix
recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
print("Recall:", recall)

# Feature Importance
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_imp)

# Print classification report
print(classification_report(y_test, y_pred_custom_threshold, zero_division=1))

# F1 Score
f1 = f1_score(y_test, y_pred_custom_threshold, pos_label='Yes')
print(f"F1 Score: {f1}")

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()


#Visualizing

single_tree = clf.estimators_[0]

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(single_tree, 
          filled=True, 
          rounded=True, 
          class_names=['No', 'Yes'],
          feature_names=X.columns,
          max_depth=3)  
plt.show()




# In[4]:


import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
import sqlite3
from sklearn.tree import plot_tree

# Connect to the SQLite database and load the DataFrame
conn = sqlite3.connect('/Users/sateeshkumarpuri/Documents/Result_Database2.db')
query = "SELECT * FROM ResultTable"
df = pd.read_sql(query, conn)
conn.close()

# Selecting specific columns for training
X = df[['ProjectDescription', 'ProjectNumber', 
        'District', 'CESTAMT', 
        'ABI_FY2014_ItemCode']]
Y = df['PotentialCO']

# Encode categorical variables
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)

# Do your train-test split on the original data
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.35, random_state=1234567)

# Create Random Forest Classifier
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=10000, class_weight={'Yes': 1, 'No': 1.0})

# Train the Classifier
clf = clf.fit(X_train, y_train)

# Get predicted probabilities
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Set custom threshold
threshold = 0.5
y_pred_custom_threshold = ['Yes' if prob > threshold else 'No' for prob in y_pred_proba]

# Confusion matrix
conf_matrix = metrics.confusion_matrix(y_test, y_pred_custom_threshold)
print("Confusion Matrix with Custom Threshold:\n", conf_matrix)

# Compute recall from the confusion matrix
recall = conf_matrix[1][1] / (conf_matrix[1][1] + conf_matrix[1][0])
print("Recall:", recall)

# Feature Importance
feature_imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
print(feature_imp)

# Print classification report
print(classification_report(y_test, y_pred_custom_threshold, zero_division=1))

# F1 Score
f1 = f1_score(y_test, y_pred_custom_threshold, pos_label='Yes')
print(f"F1 Score: {f1}")

# Feature Importance Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.show()

# Visualizing

single_tree = clf.estimators_[0]

from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(single_tree, 
          filled=True, 
          rounded=True, 
          class_names=['No', 'Yes'],
          feature_names=X.columns,
          max_depth=3)  
plt.show()

# Set the figure size
plt.figure(figsize=(20,10))
# Plot the single tree
plot_tree(single_tree, 
          filled=True, 
          rounded=True, 
          class_names=['No', 'Yes'],
          feature_names=X.columns,
          max_depth=3)  

# Save the figure
plt.savefig('/Users/sateeshkumarpuri/Documents/decision_tree.jpg', format='jpg', bbox_inches='tight', dpi = 400)
plt.close()


# In[6]:


import sqlite3
import math
import Levenshtein

def get_distinct_csj_and_project_numbers_from_database():
    # Connect to the database
    connection = sqlite3.connect("/Users/sateeshkumarpuri/Documents/Result_Database2.db")
    cursor = connection.cursor()
    
    # Query to fetch distinct CSJ and their associated project numbers
    cursor.execute("SELECT DISTINCT CSJ, ProjectNumber FROM ResultTable")
    data = cursor.fetchall()
    
    # Close the connection
    connection.close()
    
    return data

def hamming_distance(str1, str2):
    """Calculate the Hamming distance between two strings of equal length."""
    if len(str1) != len(str2):
        raise ValueError("Strings must be of the same length")
    return sum(ch1 != ch2 for ch1, ch2 in zip(str1, str2))


def calculate_similarity_scores_for_all_csj(keyword, project_data):
    results = []

    for csj, project_number in project_data:
        # Calculate Levenshtein distance
        distance = Levenshtein.distance(keyword, project_number)
        
        # Calculate similarity (normalized)
        max_len = max(len(keyword), len(project_number))
        similarity = 1 - (distance / max_len)
        results.append((csj, similarity))
    
    # Sort the results by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)

    return results

if __name__ == "__main__":
    project_data = get_distinct_csj_and_project_numbers_from_database()
    
    keyword = input("Please provide the project number for comparison: ")
    
    similarity_results = calculate_similarity_scores_for_all_csj(keyword, project_data)
    
    # Display in table format
    print("\nDistinct CSJ with Similarity Scores:")
    print("====================================")
    print("| CSJ       | Similarity Score |")
    print("|-----------|------------------|")
    for csj, score in similarity_results:
        print(f"| {csj:<9} | {score:.3f}            |")


# In[22]:


# Project Description Similarity Score

import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_distinct_csj_and_project_descriptions_from_database():
    # Connect to the database
    connection = sqlite3.connect("/Users/sateeshkumarpuri/Documents/Result_Database2.db")
    cursor = connection.cursor()
    
    # Query to fetch distinct CSJ and their associated project descriptions
    cursor.execute("SELECT DISTINCT CSJ, ProjectDescription FROM ResultTable")
    data = cursor.fetchall()
    
    # Close the connection
    connection.close()
    
    return data

def calculate_similarity_score(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity between user input and each description in the database
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Sort the results by similarity score in descending order
    results.sort(key=lambda x: x[1], reverse=True)
    
    return results

if __name__ == "__main__":
    project_data = get_distinct_csj_and_project_descriptions_from_database()
    
    project_scope = input("Please provide the project scope for comparison: ")
    
    similarity_results = calculate_similarity_score(project_scope, project_data)
    
    # Display in table format
    print("\nDistinct CSJ with similarity scores:")
    print("==================================================")
    print("| CSJ       | Similarity Score |")
    print("|-----------|------------------|")
    for csj, score in similarity_results:
        print(f"| {csj:<9} | {score:.2f}            |")



# In[ ]:


import sqlite3
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_distinct_csj_and_project_descriptions_from_database():
    # Connect to the database
    connection = sqlite3.connect("/Users/sateeshkumarpuri/Documents/Result_Database2.db")
    cursor = connection.cursor()
    
    # Query to fetch distinct CSJ and their associated project descriptions
    cursor.execute("SELECT DISTINCT CSJ, ProjectDescription FROM ResultTable")
    data = cursor.fetchall()
    
    # Close the connection
    connection.close()
    
    return data

def calculate_similarity_score(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity between user input and each description in the database
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, desc, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Sort the results by similarity score in descending order
    results.sort(key=lambda x: x[2], reverse=True)
    
    return results

if __name__ == "__main__":
    project_data = get_distinct_csj_and_project_descriptions_from_database()
    
    project_scope = input("Please provide the project scope for comparison: ")
    
    similarity_results = calculate_similarity_score(project_scope, project_data)
    
    # Display in table format
    print("\nDistinct CSJ with similarity scores and descriptions:")
    print("========================================================================")
    print("| CSJ       | Similarity Score | Project Description                     |")
    print("|-----------|------------------|-----------------------------------------|")
    for csj, desc, score in similarity_results:
        # Truncate the description if it is too long for display
        truncated_desc = (desc[:45] + '..') if len(desc) > 45 else desc
        print(f"| {csj:<9} | {score:.2f}            | {truncated_desc:<45} |")


# In[ ]:


import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import great_circle

# Predefined city coordinates
cities_coords = {
    "Austin": (30.2672, -97.7431),
    "Dallas": (32.7767, -96.7970),
    "Houston": (29.7604, -95.3698),
    "Abilene": (32.4487, -99.7331),
    "Amarillo": (35.221997, -101.831297),
    "Atlanta": (33.113743, -94.164341),
    "Beaumont": (30.0802, -94.1266),
    "Brownwood": (31.7093, -98.9912),
    "Bryan": (30.6744, -96.3698),
    "Childress": (34.4265, -100.2041),
    "Corpus Christi": (27.8006, -97.3964),
    "El Paso": (31.7619, -106.4850),
    "Fort Worth": (32.7555, -97.3308),
    "Laredo": (27.5306, -99.4803),
    "Lubbock": (33.5779, -101.8552),
    "Lufkin": (31.3382, -94.7291),
    "Odessa": (31.8457, -102.3676),
    "Paris": (33.6609, -95.5555),
    "Pharr": (26.1948, -98.1836),
    "San Angelo": (31.4638, -100.4370),
    "San Antonio": (29.4241, -98.4936),
    "Tyler": (32.3513, -95.3011),
    "Waco": (31.5493, -97.1467),
    "Wichita Falls": (33.9137, -98.4934),
    "Yoakum": (29.2884, -97.1502)
}
# Function to fetch coordinates for a given city
def get_coordinates(city):
    geolocator = Nominatim(user_agent="myUniqueAppName")
    location = geolocator.geocode(city + ", Texas")
    return (location.latitude, location.longitude) if location else None

# Connect to the SQLite database
conn = sqlite3.connect('/Users/sateeshkumarpuri/Documents/Result_Database2.db')  
query = "SELECT * FROM ResultTable" 
result_table = pd.read_sql_query(query, conn)
conn.close()

# User input for location
user_city = input("Enter your location in Texas: ")
user_coords = get_coordinates(user_city)


# Calculate distances
distances = [great_circle(user_coords, cities_coords.get(city, (0, 0))).kilometers for city in result_table['City']]

# Calculate the maximum distance for normalization
max_distance = max(distances)

# Calculate and store scores
result_table['Score'] = [1 - (d / max_distance) if max_distance != 0 else 0 for d in distances]

# Normalize scores
result_table['NormalizedScore'] = result_table['Score'] / result_table['Score'].max()

# Drop duplicates based on 'CSJ'
result_table = result_table.drop_duplicates('CSJ')

# If you want to sort by the NormalizedScore
result_table = result_table.sort_values(by='NormalizedScore', ascending=False)

# Print the result
print(f"All Distinct CSJ with corresponding cities and normalized scores for {user_city}")

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(result_table[['CSJ', 'City', 'NormalizedScore']])




# In[16]:


#Amount Similarity Score

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


# Present the table
print("CSJ\t\tNormalized Score")
print("-----------------------------")
for score, csj in sorted(normalized_scores, key=lambda x: x[0], reverse=True):
    print(f"{csj}\t\t{score:.10f}")

# Close the database connection
conn.close()



# In[ ]:


import sqlite3
import pandas as pd

def get_weighted_score(database_path, csj, item_codes):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    total_score = 0

    for item_code in item_codes:
        cursor.execute(f"""SELECT TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT 
                           FROM ResultTable 
                           WHERE CSJ = ? AND TxDOTEE_ItemCode = ?""", (csj, item_code))
        result = cursor.fetchone()

        if result:
            quantity, engineer_estimate, cestamt = result
            if cestamt != 0:
                total_score += (quantity * engineer_estimate) / cestamt

    conn.close()
    return total_score

def get_similarity_score_for_all_CSJs(database_path, scope_keywords, location_keywords, item_codes):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT CSJ, ProjectDescription, ProjectType, ProjectNumber, City, CityCode FROM ResultTable")
    csjs = cursor.fetchall()

    scores = {}
    
    for csj_data in csjs:
        csj, description, ptype, pnumber, city, city_code = csj_data
        combined_data = (description or "") + " " + (ptype or "") + " " + (pnumber or "")

        if any(keyword.lower() in (city or "").lower() or keyword.lower() in (city_code or "").lower() for keyword in location_keywords):
            if any(keyword.lower() in combined_data.lower() for keyword in scope_keywords):
                scores[csj] = get_weighted_score(database_path, csj, item_codes)

    conn.close()
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10])
    return sorted_scores

def get_potential_CO_details_for_top_CSJs(database_path, csjs):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    
    details_data = []

    for csj in csjs:
        cursor.execute(f"""SELECT ProjectNumber, ProjectDescription, HIGHWAY, City, ABI_FY2014_ItemCode, ABI_FY2014_ItemDescription, 
                                  BidQty, BidAmt, NetCOQty, NetCOAmt, CESTAMT, PotentialCO 
                           FROM ResultTable 
                           WHERE CSJ = ? AND PotentialCO = 'Yes'""", (csj,))
        result = cursor.fetchall()

        if result:
            details_data.extend(result)

    conn.close()
    return details_data

# Example usage:
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"

scope_input = input("Enter Scope keywords separated by commas (e.g., construction,bridge): ")
location_input = input("Enter Location keywords separated by commas (e.g., Dallas,Austin): ")
item_codes_input = input("Enter ITEM_CODEs separated by commas (e.g., code1,code2): ")

scope_keywords = [keyword.strip() for keyword in scope_input.split(",")]
location_keywords = [keyword.strip() for keyword in location_input.split(",")]
item_codes = [code.strip() for code in item_codes_input.split(",")]

scores = get_similarity_score_for_all_CSJs(database_path, scope_keywords, location_keywords, item_codes)

df_summary = pd.DataFrame(columns=["CSJ", "Score"])
for csj, score in scores.items():
    new_row = pd.DataFrame({"CSJ": [csj], "Score": [score]})
    df_summary = pd.concat([df_summary, new_row], ignore_index=True)

print("\nTop 10 CSJs with highest scores:")
print(df_summary)

# Ask user if they want potential CO details
choice = input("\nDo you want details of potential CO for the top 10 CSJs? (yes/no): ").lower()
if choice == 'yes':
    top_10_CSJs = list(scores.keys())
    potential_CO_details = get_potential_CO_details_for_top_CSJs(database_path, top_10_CSJs)
    
    if potential_CO_details:
        df_CO_details = pd.DataFrame(potential_CO_details, columns=["ProjectNumber", "ProjectDescription", "HIGHWAY", "City","ABI_FY2014_ItemCode", 
                                                                   "ABI_FY2014_ItemDescription", "BidQty", "BidAmt", "NetCOQty", 
                                                                   "NetCOAmt", "CESTAMT"])
        print("\nPotential CO Details for Top 10 CSJs:")
        print(df_CO_details)
    else:
        print("\nNo potential CO details found for the top 10 CSJs.")


# In[2]:


import sqlite3
import pandas as pd

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', 6)
pd.set_option('display.width', 1000)

def get_weighted_score(database_path, csj, item_codes):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    total_score = 0
    for item_code in item_codes:
        cursor.execute(f"""SELECT TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT 
                           FROM ResultTable 
                           WHERE CSJ = ? AND TxDOTEE_ItemCode = ?""", (csj, item_code))
        result = cursor.fetchone()
        if result:
            quantity, engineer_estimate, cestamt = result
            if cestamt != 0:
                total_score += (quantity * engineer_estimate) / cestamt
    conn.close()
    return total_score

def get_similarity_score_for_all_CSJs(database_path, scope_keywords, location_keywords, item_codes):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT CSJ, ProjectDescription, ProjectType, ProjectNumber, District, CityCode FROM ResultTable")
    csjs = cursor.fetchall()
    scores = {}
    potential_co_items = {}  # Store the list of potential CO item codes for each CSJ
    for csj_data in csjs:
        if any(keyword.lower() in (city or "").lower() or keyword.lower() in (city_code or "").lower() for keyword in location_keywords):
            if any(keyword.lower() in combined_data.lower() for keyword in scope_keywords):
                scores[csj] = get_weighted_score(database_path, csj, item_codes)
                
                # Get list of potential CO item codes for this CSJ
                cursor.execute(f"""SELECT ABI_FY2014_ItemCode 
                                   FROM ResultTable 
                                   WHERE CSJ = ? AND PotentialCO = 'Yes'""", (csj,))
                items = cursor.fetchall()
                potential_co_items[csj] = ', '.join([item[0] for item in items])
                
    conn.close()
    
    # Combine the score and potential CO item codes into one dictionary
    combined_scores = {csj: (score, potential_co_items.get(csj, 'None')) for csj, score in scores.items()}
    
    sorted_scores = dict(sorted(combined_scores.items(), key=lambda item: item[1], reverse=True)[:10])
    return sorted_scores


def get_similarity_score_for_all_CSJs(database_path, scope_keywords, location_keywords, item_codes):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT CSJ, ProjectDescription, ProjectType, ProjectNumber, District, CityCode FROM ResultTable")
    csjs = cursor.fetchall()
    scores = {}
    for csj_data in csjs:
        csj, description, ptype, pnumber, city, city_code = csj_data
        combined_data = (description or "") + " " + (ptype or "") + " " + (pnumber or "")
        if any(keyword.lower() in (city or "").lower() or keyword.lower() in (city_code or "").lower() for keyword in location_keywords):
            if any(keyword.lower() in combined_data.lower() for keyword in scope_keywords):
                scores[csj] = get_weighted_score(database_path, csj, item_codes)
    conn.close()
    sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:10])
    return sorted_scores

def get_potential_CO_details_for_top_CSJs(database_path, csjs):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    details_data = []
    for csj in csjs:
        cursor.execute(f"""SELECT ProjectNumber, ProjectDescription, HIGHWAY, District, CESTAMT, ABI_FY2014_ItemCode, 
                                  ABI_FY2014_ItemDescription, BidQty, BidAmt, NetCOQty, NetCOAmt 
                           FROM ResultTable 
                           WHERE CSJ = ? AND PotentialCO = 'Yes'""", (csj,))
        result = cursor.fetchall()
        if result:
            details_data.extend(result)
    conn.close()
    return details_data

# Example usage:
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
scope_input = input("Enter Scope keywords separated by commas (e.g., construction,bridge): ")
location_input = input("Enter Location keywords separated by commas (e.g., Dallas,TX123): ")
item_codes_input = input("Enter ITEM_CODEs separated by space (e.g., code1,code2): ")
scope_keywords = [keyword.strip() for keyword in scope_input.split(",")]
location_keywords = [keyword.strip() for keyword in location_input.split(",")]
item_codes = [code.strip() for code in item_codes_input.split(" ")]
scores = get_similarity_score_for_all_CSJs(database_path, scope_keywords, location_keywords, item_codes)
df_summary = pd.DataFrame(list(scores.items()), columns=["CSJ", "Score"])
print("\nTop 10 CSJs with highest scores:")
print(df_summary)
choice = input("\nDo you want details of potential CO from the top 10 CSJs? (yes/no): ").lower()
if choice == 'yes':
    top_10_CSJs = list(scores.keys())
    potential_CO_details = get_potential_CO_details_for_top_CSJs(database_path, top_10_CSJs)
    if potential_CO_details:
        # Split the data into two parts
        df_CO_details_part1 = pd.DataFrame([x[:5] for x in potential_CO_details], columns=["ProjectNumber", "ProjectDescription", "HIGHWAY", "District","CESTAMT" ])
        df_CO_details_part2 = pd.DataFrame([x[5:] for x in potential_CO_details], columns=["ABI_FY2014_ItemCode","ABI_FY2014_ItemDescription", "BidQty", "BidAmt", "NetCOQty", "NetCOAmt"])

        # Display the first part
        print("\nPotential CO Details for Top 10 CSJs (Part 1):\n")
        print(df_CO_details_part1)

        # Ask the user to continue to the next set of columns
        input("\nPress Enter to view the next set of columns...\n")

        # Display the second part
        print("\nPotential CO Details for Top 10 CSJs (Part 2):\n")
        print(df_CO_details_part2)
    else:
            print("\nNo potential CO details found for the top 10 CSJs.")


# In[3]:


import sqlite3
import pandas as pd

def get_weighted_score(row):
    quantity, engineer_estimate, cestamt = row['TxDOTEE_Quantity'], row['TxDOTEE_EngineerEstimate'], row['CESTAMT']
    if cestamt != 0:
        return (quantity * engineer_estimate) / cestamt
    return 0

def get_similarity_score_for_all_CSJs(database_path, item_codes):
    conn = sqlite3.connect(database_path)
    df = pd.read_sql_query("SELECT CSJ, TxDOTEE_ItemCode, TxDOTEE_Quantity, TxDOTEE_EngineerEstimate, CESTAMT FROM ResultTable", conn)

    # Filter rows with the specified unique item codes
    unique_item_codes = set(item_codes)
    df_filtered = df[df['TxDOTEE_ItemCode'].isin(unique_item_codes)]

    # Initialize a dictionary to hold scores and item code count for each CSJ
    scores = {csj: {'score': 0, 'item_count': 0} for csj in df_filtered['CSJ'].unique()}

    # Iterate over each CSJ
    for csj in scores.keys():
        # Filter the DataFrame for the current CSJ
        df_csj = df_filtered[df_filtered['CSJ'] == csj]

        # Initialize score sum and item count
        score_sum = 0
        item_count = 0

        for item_code in unique_item_codes:
            if item_code in df_csj['TxDOTEE_ItemCode'].values:
                row = df_csj[df_csj['TxDOTEE_ItemCode'] == item_code].iloc[0]
                score_sum += get_weighted_score(row)
                item_count += 1

        scores[csj]['score'] = score_sum
        scores[csj]['item_count'] = item_count

    conn.close()

    # Prepare data for DataFrame
    data = [(csj, round(values['score'], 4), values['item_count']) for csj, values in scores.items()]
    return data

# Inputs
database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"

# Ask the user for item codes
input_codes = input("Enter item codes separated by a space ( ): ")
item_codes = [code.strip() for code in input_codes.split(' ')]

# Get Scores
data = get_similarity_score_for_all_CSJs(database_path, item_codes)

# Convert the results to DataFrame and sort
df_summary = pd.DataFrame(data, columns=["CSJ", "Score", "ItemCount"])
sorted_df = df_summary.sort_values(by="Score", ascending=False)

# Add a Serial Number
sorted_df.reset_index(drop=True, inplace=True)
sorted_df.index += 1
sorted_df.index.name = 'Serial Number'

# Display Results
print("\nTables with cumulative scores and item counts for all distinct CSJs:")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
print(sorted_df)


# In[3]:


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


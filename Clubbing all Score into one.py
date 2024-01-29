#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sqlite3
import math
import sqlite3
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# calculate Similarity score for project number
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

def calculate_similarity_score_for_project_number(keyword, project_data):
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

def calculate_similarity_score_for_project_description(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Return results for all CSJs
    return results

if __name__ == "__main__":
    project_data_number = get_distinct_csj_and_project_numbers_from_database()
    project_data_description = get_distinct_csj_and_project_descriptions_from_database()
    
    keyword = input("Please provide the project number for comparison: ")
    project_scope = input("Please provide the project scope for comparison: ")
    
    scores_number = calculate_similarity_score_for_project_number(keyword, project_data_number)
    scores_description = calculate_similarity_score_for_project_description(project_scope, project_data_description)
    
    # Combine the scores
    combined_scores = {}
    for csj, score in scores_number:
        combined_scores[csj] = score
    
    for csj, score in scores_description:
        combined_scores[csj] = combined_scores.get(csj, 0) + score
    
        # Sort and return all results instead of just top 20
    all_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Display results
    print("\nAll Distinct CSJ with maximum combined similarity scores:")
    print("=============================================================")
    print("| CSJ       | Combined Similarity Score |")
    print("|-----------|---------------------------|")
    for csj, score in all_results:
        print(f"| {csj:<9} | {score:.4f}                     |")


# In[5]:


import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# Function definitions (unchanged from your original code)

# calculate Similarity score for project number
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

def calculate_similarity_score_for_project_number(keyword, project_data):
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

def calculate_similarity_score_for_project_description(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Return results for all CSJs
    return results


# New function for calculating normalized distance scores

def calculate_distance_scores(user_city):
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
    user_coords = get_coordinates(user_city)

    # Calculate distances
    distances = [great_circle(user_coords, cities_coords.get(city, (0, 0))).kilometers for city in result_table['City']]

    # Calculate the maximum distance for normalization
    max_distance = max(distances)

    # Calculate and store scores
    result_table['Score'] = [1 - (d / max_distance) if max_distance != 0 else 0 for d in distances]

    # Normalize scores
    result_table['NormalizedScore'] = result_table['Score'] / result_table['Score'].max()

    # Drop duplicates based on 'CSJ' and return the result
    result_table = result_table.drop_duplicates('CSJ')
    return result_table.set_index('CSJ')['NormalizedScore']

if __name__ == "__main__":
    project_data_number = get_distinct_csj_and_project_numbers_from_database()
    project_data_description = get_distinct_csj_and_project_descriptions_from_database()
    
    keyword = input("Please provide the project number for comparison: ")
    project_scope = input("Please provide the project scope for comparison: ")

    # User input for location
    user_city = input("Enter your location in Texas: ")

    # Calculate all scores
    scores_number = calculate_similarity_score_for_project_number(keyword, project_data_number)
    scores_description = calculate_similarity_score_for_project_description(project_scope, project_data_description)
    scores_distance = calculate_distance_scores(user_city)

    # Combine the scores
    combined_scores = {}
    for csj, score in scores_number:
        combined_scores[csj] = score
    
    for csj, score in scores_description:
        combined_scores[csj] = combined_scores.get(csj, 0) + score

    for csj, score in scores_distance.items():
        combined_scores[csj] = combined_scores.get(csj, 0) + score

    # Sort and return all results
    all_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Display results
    print("\nAll Distinct CSJ with maximum combined similarity scores:")
    print("=============================================================")
    print("| CSJ       | Combined Similarity Score |")
    print("|-----------|---------------------------|")
    for csj, score in all_results:
        print(f"| {csj:<9} | {score:.4f}                     |")


# In[ ]:


import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# calculate Similarity score for project number
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

def calculate_similarity_score_for_project_number(keyword, project_data):
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

def calculate_similarity_score_for_project_description(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Return results for all CSJs
    return results


# New function for calculating normalized distance scores

def calculate_distance_scores(user_city):
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
    user_coords = get_coordinates(user_city)

    # Calculate distances
    distances = [great_circle(user_coords, cities_coords.get(city, (0, 0))).kilometers for city in result_table['City']]

    # Calculate the maximum distance for normalization
    max_distance = max(distances)

    # Calculate and store scores
    result_table['Score'] = [1 - (d / max_distance) if max_distance != 0 else 0 for d in distances]

    # Normalize scores
    result_table['NormalizedScore'] = result_table['Score'] / result_table['Score'].max()

    # Drop duplicates based on 'CSJ' and return the result
    result_table = result_table.drop_duplicates('CSJ')
    return result_table.set_index('CSJ')['NormalizedScore']

# New function for calculating amount similarity scores

def calculate_amount_similarity_scores(user_amount):
    # Connect to the database
    db_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract amounts and corresponding CSJs from the database
    cursor.execute("SELECT DISTINCT CESTAMT, CSJ FROM ResultTable")
    rows = cursor.fetchall()

    # Handle no data case
    if not rows:
        return {}

    # Calculate the squared error and scores
    errors = [(user_amount - float(row[0]))**2 for row in rows]
    max_error = max(errors)
    scores = {row[1]: max_error / (error + 1e-10) for error, row in zip(errors, rows)}

    # Normalize the scores
    max_score = max(scores.values())
    normalized_scores = {csj: score / max_score for csj, score in scores.items()}

    # Close the database connection
    conn.close()

    return normalized_scores

def get_full_project_data_from_database():
    # Connect to the database
    connection = sqlite3.connect("/Users/sateeshkumarpuri/Documents/Result_Database2.db")
    cursor = connection.cursor()
    
    # Query to fetch all relevant data for the projects
    query = """
    SELECT DISTINCT CSJ, ProjectNumber, ProjectDescription, City, CESTAMT 
    FROM ResultTable
    """
    cursor.execute(query)
    data = cursor.fetchall()
    
    # Close the connection
    connection.close()
    
    return data


if __name__ == "__main__":
    

    # Fetch full project data
    full_project_data = get_full_project_data_from_database()
    # Fetch existing data
    project_data_number = get_distinct_csj_and_project_numbers_from_database()
    project_data_description = get_distinct_csj_and_project_descriptions_from_database()

    # Get user inputs for different comparisons
    keyword = input("Please provide the project number for comparison: ")
    project_scope = input("Please provide the project scope for comparison: ")
    user_city = input("Enter your location in Texas: ")
    user_amount = float(input("Please enter your project amount: "))

    # Calculate all scores
    scores_number = calculate_similarity_score_for_project_number(keyword, project_data_number)
    scores_description = calculate_similarity_score_for_project_description(project_scope, project_data_description)
    scores_distance = calculate_distance_scores(user_city)
    scores_amount = calculate_amount_similarity_scores(user_amount)

    # Combine the scores
    combined_scores = {}
    for csj, score in scores_number:
        combined_scores[csj] = score
    
    for csj, score in scores_description:
        combined_scores[csj] = combined_scores.get(csj, 0) + score

    for csj, score in scores_distance.items():
        combined_scores[csj] = combined_scores.get(csj, 0) + score

    for csj, score in scores_amount.items():
        combined_scores[csj] = combined_scores.get(csj, 0) + score


    # Combine the scores and include additional data
    combined_data = {}
    for csj, project_number, project_description, city, amount in full_project_data:
        score_number = next((score for c, score in scores_number if c == csj), 0)
        score_description = next((score for c, score in scores_description if c == csj), 0)
        score_distance = scores_distance.get(csj, 0)
        score_amount = scores_amount.get(csj, 0)
        combined_score = score_number + score_description + score_distance + score_amount
        
        combined_data[csj] = {
            'CombinedScore': combined_score,
            'ProjectNumber': project_number,
            'ProjectDescription': project_description,
            'Location': city,
            'ProjectAmount': amount
        }
    
# Sort and return all results
    all_results = sorted(combined_data.items(), key=lambda x: x[1]['CombinedScore'], reverse=True)
    
    # Display results
    print("\nAll Distinct CSJ with maximum combined similarity scores:")
    print("=================================================================================")
    print("| CSJ       | Combined Score | Project Number | Description       | Location | Amount |")
    print("|-----------|----------------|----------------|-------------------|----------|--------|")
    for csj, data in all_results:
        print(f"| {csj:<9} | {data['CombinedScore']:.4f}         | {data['ProjectNumber']}            | {data['ProjectDescription'][:20]:<20} | {data['Location']:<8} | {data['ProjectAmount']:<6} |")


# In[7]:


import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# calculate Similarity score for project number
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

def calculate_similarity_score_for_project_number(keyword, project_data):
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

def calculate_similarity_score_for_project_description(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Return results for all CSJs
    return results


# New function for calculating normalized distance scores

def calculate_distance_scores(user_city):
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
    user_coords = get_coordinates(user_city)

    # Calculate distances
    distances = [great_circle(user_coords, cities_coords.get(city, (0, 0))).kilometers for city in result_table['District']]

    # Calculate the maximum distance for normalization
    max_distance = max(distances)

    # Calculate and store scores
    result_table['Score'] = [1 - (d / max_distance) if max_distance != 0 else 0 for d in distances]

    # Normalize scores
    result_table['NormalizedScore'] = result_table['Score'] / result_table['Score'].max()

    # Drop duplicates based on 'CSJ' and return the result
    result_table = result_table.drop_duplicates('CSJ')
    return result_table.set_index('CSJ')['NormalizedScore']

# New function for calculating amount similarity scores

def calculate_amount_similarity_scores(user_amount):
    # Connect to the database
    db_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract amounts and corresponding CSJs from the database
    cursor.execute("SELECT DISTINCT CESTAMT, CSJ FROM ResultTable")
    rows = cursor.fetchall()

    # Handle no data case
    if not rows:
        return {}

    # Calculate the squared error and scores
    errors = [(user_amount - float(row[0]))**2 for row in rows]
    max_error = max(errors)
    scores = {row[1]: max_error / (error + 1e-10) for error, row in zip(errors, rows)}

    # Normalize the scores
    max_score = max(scores.values())
    normalized_scores = {csj: score / max_score for csj, score in scores.items()}

    # Close the database connection
    conn.close()

    return normalized_scores
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
    return scores

# Main execution block
if __name__ == "__main__":
    
    # Define the database path
    database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    
    # Fetch existing data
    project_data_number = get_distinct_csj_and_project_numbers_from_database()
    project_data_description = get_distinct_csj_and_project_descriptions_from_database()

    # Get user inputs for different comparisons
    keyword = input("Please provide the project number for comparison: ")
    project_scope = input("Please provide the project scope for comparison: ")
    user_city = input("Enter your location in Texas: ")
    user_amount = float(input("Please enter your project amount: "))
    input_codes = input("Enter item codes separated by a space ( ): ")
    item_codes = [code.strip() for code in input_codes.split(' ')]

    # Calculate all similarity scores
    scores_number = calculate_similarity_score_for_project_number(keyword, project_data_number)
    scores_description = calculate_similarity_score_for_project_description(project_scope, project_data_description)
    scores_distance = calculate_distance_scores(user_city)
    scores_amount = calculate_amount_similarity_scores(user_amount)

    # Calculate item code scores and counts
    item_code_data = get_similarity_score_for_all_CSJs(database_path, item_codes)

    # Combine all scores and include item counts
    combined_scores = {csj: {'combined_score': 0, 'item_count': 0} for csj, _ in project_data_number}

    # Add scores from project number similarity
    for csj, score in scores_number:
        combined_scores[csj]['combined_score'] += score

    # Add scores from project description similarity
    for csj, score in scores_description:
        combined_scores[csj]['combined_score'] += score

    # Add scores from distance similarity
    for csj, score in scores_distance.items():
        combined_scores[csj]['combined_score'] += score

    # Add scores from amount similarity
    for csj, score in scores_amount.items():
        combined_scores[csj]['combined_score'] += score

    # Add item code similarity scores and counts
    for csj, values in item_code_data.items():
        combined_scores[csj]['combined_score'] += values['score']
        combined_scores[csj]['item_count'] = values['item_count']

    # Sort the combined scores by combined score and item count
    sorted_scores = sorted(combined_scores.items(), key=lambda x: (x[1]['combined_score'], x[1]['item_count']), reverse=True)

    # Display only top N results
    top_n = 10  # Adjust the number to display more or fewer results
    print("\nTop CSJs based on combined similarity scores:")
    for i, (csj, data) in enumerate(sorted_scores[:top_n], 1):
        print(f"{i}. CSJ: {csj}, Combined Score: {data['combined_score']:.4f}, Item Count: {data['item_count']}")


# In[6]:


import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', None)  # Show all rows

# calculate Similarity score for project number
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

def calculate_similarity_score_for_project_number(keyword, project_data):
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

def calculate_similarity_score_for_project_description(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Return results for all CSJs
    return results


# New function for calculating normalized distance scores

def calculate_distance_scores(user_city):
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
    user_coords = get_coordinates(user_city)

    # Calculate distances
    distances = [great_circle(user_coords, cities_coords.get(city, (0, 0))).kilometers for city in result_table['District']]

    # Calculate the maximum distance for normalization
    max_distance = max(distances)

    # Calculate and store scores
    result_table['Score'] = [1 - (d / max_distance) if max_distance != 0 else 0 for d in distances]

    # Normalize scores
    result_table['NormalizedScore'] = result_table['Score'] / result_table['Score'].max()

    # Drop duplicates based on 'CSJ' and return the result
    result_table = result_table.drop_duplicates('CSJ')
    return result_table.set_index('CSJ')['NormalizedScore']

# New function for calculating amount similarity scores

def calculate_amount_similarity_scores(user_amount):
    # Connect to the database
    db_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract amounts and corresponding CSJs from the database
    cursor.execute("SELECT DISTINCT CESTAMT, CSJ FROM ResultTable")
    rows = cursor.fetchall()

    # Handle no data case
    if not rows:
        return {}

    # Calculate the squared error and scores
    errors = [(user_amount - float(row[0]))**2 for row in rows]
    max_error = max(errors)
    scores = {row[1]: max_error / (error + 1e-10) for error, row in zip(errors, rows)}

    # Normalize the scores
    max_score = max(scores.values())
    normalized_scores = {csj: score / max_score for csj, score in scores.items()}

    # Close the database connection
    conn.close()

    return normalized_scores
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
    return scores

# Main execution block
if __name__ == "__main__":
    
    # Define the database path
    database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    
    # Fetch existing data
    project_data_number = get_distinct_csj_and_project_numbers_from_database()
    project_data_description = get_distinct_csj_and_project_descriptions_from_database()

    # Get user inputs for different comparisons
    keyword = input("Please provide the project number for comparison: ")
    project_scope = input("Please provide the project scope for comparison: ")
    user_city = input("Enter your location in Texas: ")
    user_amount = float(input("Please enter your project amount: "))
    input_codes = input("Enter item codes separated by a space ( ): ")
    item_codes = [code.strip() for code in input_codes.split(' ')]

    # Calculate all similarity scores
    scores_number = calculate_similarity_score_for_project_number(keyword, project_data_number)
    scores_description = calculate_similarity_score_for_project_description(project_scope, project_data_description)
    scores_distance = calculate_distance_scores(user_city)
    scores_amount = calculate_amount_similarity_scores(user_amount)

    # Calculate item code scores and counts
    item_code_data = get_similarity_score_for_all_CSJs(database_path, item_codes)

    # Combine all scores and include item counts
    combined_scores = {csj: {'combined_score': 0, 'item_count': 0} for csj, _ in project_data_number}

    # Add scores from project number similarity
    for csj, score in scores_number:
        combined_scores[csj]['combined_score'] += score

    # Add scores from project description similarity
    for csj, score in scores_description:
        combined_scores[csj]['combined_score'] += score

    # Add scores from distance similarity
    for csj, score in scores_distance.items():
        combined_scores[csj]['combined_score'] += score

    # Add scores from amount similarity
    for csj, score in scores_amount.items():
        combined_scores[csj]['combined_score'] += score

    # Add item code similarity scores and counts
    for csj, values in item_code_data.items():
        combined_scores[csj]['combined_score'] += values['score']
        combined_scores[csj]['item_count'] = values['item_count']

    # Sort the combined scores first by item count and then by combined score
    sorted_by_item_count = sorted(combined_scores.items(), key=lambda x: (x[1]['item_count'], x[1]['combined_score']), reverse=True)

    # Display only top 10 results based on item count
    print("\nTop 10 CSJs based on item count:")
    for i, (csj, data) in enumerate(sorted_by_item_count[:10], 1):
        print(f"{i}. CSJ: {csj}, Item Count: {data['item_count']}, Combined Score: {data['combined_score']:.4f}")


# In[13]:


import sqlite3
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.distance import great_circle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein



# calculate Similarity score for project number
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

def calculate_similarity_score_for_project_number(keyword, project_data):
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

def calculate_similarity_score_for_project_description(project_scope, project_data):
    vectorizer = TfidfVectorizer()
    
    descriptions = [desc for csj, desc in project_data]
    descriptions.append(project_scope)  # Append the user's input to the end for vectorization
    
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    
    # Compute cosine similarity
    cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    results = [(csj, score) for (csj, desc), score in zip(project_data, cosine_similarities[0])]

    # Return results for all CSJs
    return results


# New function for calculating normalized distance scores

def calculate_distance_scores(user_city):
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
    user_coords = get_coordinates(user_city)

    # Calculate distances
    distances = [great_circle(user_coords, cities_coords.get(city, (0, 0))).kilometers for city in result_table['District']]

    # Calculate the maximum distance for normalization
    max_distance = max(distances)

    # Calculate and store scores
    result_table['Score'] = [1 - (d / max_distance) if max_distance != 0 else 0 for d in distances]

    # Normalize scores
    result_table['NormalizedScore'] = result_table['Score'] / result_table['Score'].max()

    # Drop duplicates based on 'CSJ' and return the result
    result_table = result_table.drop_duplicates('CSJ')
    return result_table.set_index('CSJ')['NormalizedScore']

# New function for calculating amount similarity scores

def calculate_amount_similarity_scores(user_amount):
    # Connect to the database
    db_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Extract amounts and corresponding CSJs from the database
    cursor.execute("SELECT DISTINCT CESTAMT, CSJ FROM ResultTable")
    rows = cursor.fetchall()

    # Handle no data case
    if not rows:
        return {}

    # Calculate the squared error and scores
    errors = [(user_amount - float(row[0]))**2 for row in rows]
    max_error = max(errors)
    scores = {row[1]: max_error / (error + 1e-10) for error, row in zip(errors, rows)}

    # Normalize the scores
    max_score = max(scores.values())
    normalized_scores = {csj: score / max_score for csj, score in scores.items()}

    # Close the database connection
    conn.close()

    return normalized_scores
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
    return scores


# Adjust display settings for pandas DataFrame
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', 1000)  # Set display width for wide tables
pd.set_option('display.max_colwidth', None)  # Show full width of content in columns
pd.set_option('display.colheader_justify', 'center')  # Center-align column headers
pd.set_option('display.precision', 2)  # Set precision for floating point numbers


# Main execution block
if __name__ == "__main__":
    
    # Define the database path
    database_path = "/Users/sateeshkumarpuri/Documents/Result_Database2.db"
    
    # Fetch existing data
    project_data_number = get_distinct_csj_and_project_numbers_from_database()
    project_data_description = get_distinct_csj_and_project_descriptions_from_database()

    # Get user inputs for different comparisons
    keyword = input("Please provide the project number for comparison: ")
    project_scope = input("Please provide the project scope for comparison: ")
    user_city = input("Enter your location in Texas: ")
    user_amount = float(input("Please enter your project amount: "))
    input_codes = input("Enter item codes separated by a space ( ): ")
    item_codes = [code.strip() for code in input_codes.split(' ')]

    # Calculate all similarity scores
    scores_number = calculate_similarity_score_for_project_number(keyword, project_data_number)
    scores_description = calculate_similarity_score_for_project_description(project_scope, project_data_description)
    scores_distance = calculate_distance_scores(user_city)
    scores_amount = calculate_amount_similarity_scores(user_amount)

    # Calculate item code scores and counts
    item_code_data = get_similarity_score_for_all_CSJs(database_path, item_codes)

    # Combine all scores and include item counts
    combined_scores = {csj: {'combined_score': 0, 'item_count': 0} for csj, _ in project_data_number}

    # Add scores from project number similarity
    for csj, score in scores_number:
        combined_scores[csj]['combined_score'] += score

    # Add scores from project description similarity
    for csj, score in scores_description:
        combined_scores[csj]['combined_score'] += score

    # Add scores from distance similarity
    for csj, score in scores_distance.items():
        combined_scores[csj]['combined_score'] += score

    # Add scores from amount similarity
    for csj, score in scores_amount.items():
        combined_scores[csj]['combined_score'] += score

    # Add item code similarity scores and counts
    for csj, values in item_code_data.items():
        combined_scores[csj]['combined_score'] += values['score']
        combined_scores[csj]['item_count'] = values['item_count']

    # Sort the combined scores first by combined score and then by item count
    sorted_scores = sorted(combined_scores.items(), key=lambda x: (x[1]['combined_score'], x[1]['item_count']), reverse=True)

    # Display only top 10 results based on combined score
    top_10_csjs = [(csj, data['combined_score']) for csj, data in sorted_scores[:10]]
    df_top_csjs = pd.DataFrame(top_10_csjs, columns=['CSJ', 'Score'])
    print("\nTop 10 CSJs with highest scores:")
    print(df_top_csjs)

    # Prompt user to display details of top 10 CSJs
    if input("\nDo you want details of the top 10 CSJs? (yes/no): ").lower() == 'yes':
        conn = sqlite3.connect(database_path)
        cursor = conn.cursor()
        details_data = []
        for csj, _ in top_10_csjs:
            cursor.execute("SELECT ProjectNumber, ProjectDescription, ProjectType, HIGHWAY, District, CESTAMT FROM ResultTable WHERE CSJ = ?", (csj,))
            details_data.append(cursor.fetchone())
        conn.close()

        df_details = pd.DataFrame(details_data, columns=["ProjectNumber", "ProjectDescription", "ProjectType", "HIGHWAY", "District", "CESTAMT"])
        print("\nDetails of the top 10 CSJs:")
        print(df_details.to_string(index=False))



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

        


# In[ ]:





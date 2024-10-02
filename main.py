# Cell 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors

# Cell 2: Load and Preprocess the Data
# Load the dataset
book_data = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv')

# Display the first few rows of the dataset
book_data.head()

# Clean the dataset (removing duplicates, NaN values, etc.)
book_data.drop_duplicates(subset='title', keep='first', inplace=True)
book_data = book_data.dropna(subset=['title'])

# Load the ratings data
ratings_data = pd.read_csv('https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv')

# Display the first few rows of the ratings dataset
ratings_data.head()

# Merge book and ratings data
merged_data = pd.merge(ratings_data, book_data, on='book_id')

# Remove users with less than 200 ratings and books with less than 100 ratings
ratings_count = merged_data.groupby('user_id').size()
users_to_keep = ratings_count[ratings_count >= 200].index
filtered_data = merged_data[merged_data['user_id'].isin(users_to_keep)]

book_ratings_count = filtered_data['title'].value_counts()
books_to_keep = book_ratings_count[book_ratings_count >= 100].index
filtered_data = filtered_data[filtered_data['title'].isin(books_to_keep)]

# Pivot the data to create a user-item matrix
user_item_matrix = filtered_data.pivot_table(index='user_id', columns='title', values='rating').fillna(0)

# Cell 3: Create the Recommendation Model
# Fit Nearest Neighbors model
knn = NearestNeighbors(n_neighbors=6, algorithm='ball_tree')
knn.fit(user_item_matrix.T)  # Transpose to fit on items

# Function to get recommendations
def get_recommends(book_title):
    if book_title not in user_item_matrix.columns:
        return f"{book_title} not found in the dataset."
    
    distances, indices = knn.kneighbors(user_item_matrix[book_title].values.reshape(1, -1), n_neighbors=6)
    
    recommendations = []
    for i in range(1, len(distances.flatten())):
        recommendations.append([user_item_matrix.columns[indices.flatten()[i]], distances.flatten()[i]])
    
    return [book_title, recommendations]

# Cell 4: Test the Recommendation Function
# Test the function
recommended_books = get_recommends("The Queen of the Damned (Vampire Chronicles (Paperback))")
print(recommended_books)
   
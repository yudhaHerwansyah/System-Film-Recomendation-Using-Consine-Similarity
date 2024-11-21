from flask import Flask, render_template, request
import pandas as pd
import pickle
import re

app = Flask(__name__)

# Load models from .pkl files
with open('count_vectorizer.pkl', 'rb') as file:
    tf = pickle.load(file)

with open('cosine_similarity.pkl', 'rb') as file:
    cosine_sim = pickle.load(file)

# Load the modified movie dataset (with cleaned titles and years)
movies_meta_data = pd.read_csv('modified_movies.csv')

# Function to clean the input movie title by removing the year in parentheses
def clean_title(title):
    return re.sub(r'\(\d{4}\)', '', title).strip()

# Create indices for the movie titles (cleaned titles)
# Convert movie titles to lowercase to handle case-insensitive matching
movies_meta_data['cleaned_title'] = movies_meta_data['judul'].apply(lambda x: clean_title(x).lower())
indices = pd.Series(movies_meta_data.index, index=movies_meta_data['cleaned_title']).drop_duplicates()

def movie_recommendations(title, cosine_sim=cosine_sim):
    # Clean the input title to remove the year and convert it to lowercase
    clean_input_title = clean_title(title).lower()
    
    # Check if the cleaned movie title is in the dataset
    if clean_input_title not in indices.index:
        return "Sorry, The Movie You Entered Is Not In Our Database."

    idx = indices[clean_input_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get the top 10 most similar movies
    movie_indices = [i[0] for i in sim_scores]
    
    # Retrieve the titles, years, and genres of the recommended movies
    recommendations = []
    for movie_index in movie_indices:
        movie_title = movies_meta_data['judul'].iloc[movie_index]
        movie_year = str(movies_meta_data['year'].iloc[movie_index])  # Convert to string
        movie_genre = movies_meta_data['genre'].iloc[movie_index]
        recommendations.append({'title': movie_title, 'year': movie_year, 'genre': movie_genre})
    
    return recommendations

@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    error_message = ""
    
    if request.method == 'POST':
        movie_title = request.form.get('movie_title')
        recommendations = movie_recommendations(movie_title)
        
        # If no recommendations found, it means there's an error message
        if isinstance(recommendations, str):
            error_message = recommendations
            recommendations = []  # No recommendations if there's an error

    return render_template('index.html', recommendations=recommendations, error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
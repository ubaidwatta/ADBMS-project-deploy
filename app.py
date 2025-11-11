import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from tabulate import tabulate
import streamlit as st  # Import streamlit

# --- Data Loading and Processing (Cached) ---
# @st.cache_data runs this function ONCE and saves the result.
# This means your app will be fast and won't reload the data every time.
@st.cache_data
def load_data_and_build_model():
    """
    Loads data, builds the user-movie matrix, and computes the
    similarity dataframe. Returns all necessary objects.
    """
    print("Loading data and building model...")
    # Load data
    try:
        movies = pd.read_csv('movies.csv')
        ratings = pd.read_csv('ratings.csv')
    except FileNotFoundError:
        st.error("Error: 'movies.csv' or 'ratings.csv' not found.")
        st.stop()

    # Create the user-movie rating matrix
    user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

    # Compute the cosine similarity matrix between users
    similarity_matrix = cosine_similarity(user_movie_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=user_movie_matrix.index, columns=user_movie_matrix.index)

    # Create a mapping from custom 1-based ID to original userId
    original_user_ids = user_movie_matrix.index.tolist()
    
    print("Model build complete.")
    return movies, ratings, similarity_df, original_user_ids

# --- Recommendation Function ---
def get_recommendations(custom_user_id, genre, num_recommendations=5):
    """
    Generates movie recommendations for a user.
    (This function is now separate from the data loading)
    """
    
    try:
        original_user_id = original_user_ids[custom_user_id - 1]
    except IndexError:
        st.error(f"Error: User ID {custom_user_id} is out of range. Please enter a value between 1 and {len(original_user_ids)}.")
        return pd.DataFrame() # Return empty DataFrame on error

    if genre:
        filtered_movies = movies[movies['genres'].str.contains(genre, case=False, na=False)]
    else:
        filtered_movies = movies
    
    # Find similar users
    similar_users = similarity_df[original_user_id].sort_values(ascending=False)[1:num_recommendations + 1].index
    
    # Get movies rated by similar users
    recommended_movies = ratings[ratings['userId'].isin(similar_users)]
    
    # Filter for genre and get top recommendations
    movie_recommendations = (recommended_movies[recommended_movies['movieId'].isin(filtered_movies['movieId'])]
                             .groupby('movieId')['rating'].mean()
                             .sort_values(ascending=False)
                             .head(num_recommendations))
    
    # Get movie details
    recommended_movies_details = pd.merge(movie_recommendations.reset_index(), movies, on='movieId')
    
    return recommended_movies_details[['title', 'genres', 'rating']]

# --- Load Data (runs only once) ---
movies, ratings, similarity_df, original_user_ids = load_data_and_build_model()

# --- Streamlit UI ---
st.title("🎬 Movie Recommendation System")
st.markdown("Based on your User ID, this app will find similar users and recommend movies they liked.")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    # Number input for User ID
    custom_user_id = st.number_input(
        f"Enter your user ID (1-{len(original_user_ids)}):",
        min_value=1,
        max_value=len(original_user_ids),
        value=1,
        step=1
    )

with col2:
    # Text input for Genre
    genre_input = st.text_input("Enter a genre (e.g., Action) or leave blank for any:")
    
    # Clean up genre input
    genre = genre_input.strip() if genre_input.strip() else None

# --- Generate Recommendations ---
if st.button("Get Recommendations"):
    with st.spinner("Finding movies for you..."):
        recommended_movies = get_recommendations(
            custom_user_id=custom_user_id, 
            genre=genre
        )

        if not recommended_movies.empty:
            st.subheader("Here are your recommended movies:")
            
            # Reset index to start from 1 for display
            recommended_movies.index = range(1, len(recommended_movies) + 1)
            
            # Use st.dataframe to display the table
            st.dataframe(recommended_movies, use_container_width=True)
        else:
            st.warning(f"No recommendations found for user {custom_user_id}" + (f" in the '{genre}' genre." if genre else "."))
import numpy as np
import pandas as pd
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

dir = "ml-latest//"
movies = pd.read_csv(dir + "movies.csv")
tags = pd.read_csv(dir + "tags.csv")
tags = tags.sample(n=500000, random_state=42)
ratings = pd.read_csv(dir + "ratings.csv", nrows =1000)


n = 5 #This will be the number of suggestions we will see in the end result 


#________________________ CLEANING UP ____________________________

# and now we remove those movies from the list as they won't be useful for the recommendation based on genres
movies = movies[movies["genres"] != "(no genres listed)"].reset_index(drop=True)

#I'll use regex to remove the years from the title and instead add them to a new column so they can be used for advanced filtering.
#This means in case of duplicates we'll have to ask the user to choose between several options
movies["year"] = movies["title"].str.extract(r"\((\d{4})\)$")
#This part extracted the part in parentheses containing 4 digits, and added it to a new column
movies["title"] = movies["title"].str.replace(r"\(\d{4}\)$", "", regex=True).str.strip()
#This line removes the years from the title with the same parameters as the previous
#regex=True tells pandas to interpret \(\d{4}\)$ as pattern instead of literal string of text
#str.strip() removes any leftover whitespace after removal

# since the genres in the database are divided by | and pandas expects a space we will adapt them
movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

#From tags we'll remove the unnecessary columns
tags = tags.drop(columns=["userId","timestamp"])
#Now that we only have the data we need, we can merge the rows with the same movieID to collect all tags in a single column
tags = tags.dropna(subset=["tag"])
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()

#__________________________MERGING____________________________
movies = movies.merge(tags_grouped, on="movieId",how="left")
#Now we have a column for genres and one for tags. We want to merge them 
#First we need to take care of the NaN values by replacing them with an empty string
movies["tag"] = movies["tag"].fillna("")
movies["combined"] = movies["genres"]+ " " + movies["tag"]

movies = movies.drop("tag", axis=1)

#______________________TF-IDF VOCABULARY________________________
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(movies["combined"])

#_______________________FUNCTIONS_______________________________
def get_index (movie_title):
    result = movies[movies["title"]== movie_title]
    if result.empty:
        print(f"Movie {movie_title} not in the database")
        return None
    return result.index[0]

def get_vector(index, tfidf_matrix):
    row = tfidf_matrix[index].toarray()
    return row

def get_similarity (index, tfidf_matrix):
    row = get_vector(index, tfidf_matrix)
    similarity = cosine_similarity(row, tfidf_matrix)
    return similarity

def recommend_movies(movie_title, tfidf_matrix):
    movie_idx = get_index(movie_title)
    if movie_idx is None:
        return

    similarity = get_similarity(movie_idx, tfidf_matrix)
    sim = similarity[0]

    sorted_sim = sim.argsort()[::-1]
    matches = []

#This system will make sure the title in our search is not included in the results even in case it's not in first position
    for idx in sorted_sim:
        if len(matches) == n:
            break
        if movies["title"].iloc[idx] != movie_title:
            matches.append(movies["title"].iloc[idx])

    return matches

if __name__ == "__main__":
    parser= argparse.ArgumentParser()
    parser.add_argument("movie", type=str, help="Movie title to get recommendations for")
    args = parser.parse_args()

    print(recommend_movies(args.movie, tfidf_matrix))
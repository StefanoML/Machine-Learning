# Report on the Movie-Recommendation System

## 1 - Introduction:

The assignment required the construction of a movie recommendation system using the data from movielens.

Given a movie title as input, the system recommends 5 similar movies based on their features.

Several methods of execution are suggested, I chose to go for a TF-TDF approach combining movies-genres and user generated tags, with cosine similarity to measure the distance between movies. Choosing genres only would produce very poor results with only 19 genres for over 79000 movies.  Many movies share the same genres combinations but have very little in common, especially those marked as Drama, the most common genre. 

##### Instructions: 

The .py file will look for the "ml-latest" folder in the same directory in which it's placed. So make sure to set the right directory, the files movies.csv and tags.csv should be inside the "ml-latest" folder.

The system runs as a command-line script with the title of the input movie as an argument.

Example:

```python
python movie_recommendation.py "Jumanji"
```

## 2 - Dataset:

Of the files contained in the folder I used:

**movies.csv**: 86500 movies with  title, genres and movie ID .

The genres were separated by | and I changed that to space-separated strings for TF-IDF.

The titles of movies also included the years for disambiguation, but I thought that might become problematic when searching for maually input titles, so I decided to remove the year from the title and move it instead to a separate column. That will allow for filtering options in a future more advanced version, and manual disambiguation (When an input that has several matches is found the system will ask the user to choose between the available options).


**tags.csv**: This file was much bigger, over 1 million user generated free text tags, presented one per row. After several attempts I landed on a sample of 500.000 rows taking into consideration memory usage and coverage. The whole dataset could be too big for some computers and a small dataset would produce poor recommendations. 

I decided to use tags together with genres as they complete eachother in a way in which genres cover a more generic description while tags add more detail to it. 

---



I didn't use the other datasets as **ratings.csv** contained over 30million rows and would have needed to be reduced quite a lot and its use was a bit beyond my scope. The other files were not part of the assignment. 

---

I did some data exploration for the general shape of the data in the notebook version of the program, looking at missing data, shapes etc. In a separate file I did some deeper analysis on the actual information contained in the data, based on genres, movies and years, with some plots and charts to present the data and some personal observations.

Some conclusions from that observation:


* Drama dominates genre frequency, being a broad descriptor applicable to most films
* Movie production is heavily skewed toward recent decades, with a visible drop in 2021-2023 due to COVID and industry strikes
* Approximately 64% of movies have at least one user tag, leaving 36% described by genres alone
* Sci-Fi is the most frequent tag, reflecting both a production boom in the genre and an engaged online fanbase
* Tag density follows genre frequency, with Drama and Comedy having the richest tag coverage


## 3 - Methods:

##### TF-IDF Vectorization:

TF-IDF converts text into numerical vectors where each value represents how important a word is to a specific document relative to the whole collection. Common words across all movies get low scores, distinctive words get high scores.

Each movie is represented as a combined string of its genres and user tags. This combined text is fed into a TF-IDF vectorizer, producing a matrix of shape (79,477 × vocabulary size) where each row is a movie and each column is a word weight.

##### Cosine Similarity:

To measure similarity between movies, cosine similarity is used. Rather than measuring the distance between vectors, it measures the angle between them (two movies with similar content will point in the same direction in the feature space regardless of how many tags they have). Values range from 0 (completely different) to 1 (identical).

Given an input movie, its TF-IDF vector is compared against all other movies in the matrix. The resulting similarity scores are sorted and the top results are returned as recommendations.

##### Why TF-IDF over One-Hot-Encoding:

A simpler baseline would be one-hot encoding of genres with KNN cosine similarity. However with only 19 unique genres, many movies share identical genre combinations and produce similarity scores of 1.0,  making recommendations essentially random among a large pool of identical vectors. Usinfg user tags with TF-IDF produces a much more unique movie classification which gives more accurate results. 

## 4 - Implementation Choices:

##### Year Extraction from Title:

Movie titles in the dataset include the release year in parentheses, for example "Toy Story (1995)", which could make the input search problematic. Rather than simply removing these, the years were extracted into a separate column. This preserves the information for potential future filtering by year, and also makes it possible to disambiguate movies that share the same title. "Moana", for example, exists as three different films from 1926, 2009 and 2016. Without the year as a reference point, the system has no way to tell them apart. The current implementation returns the first match found when duplicates exist. This is a known limitation. A future version would detect multiple matches and ask the user to choose a version before proceeding.

##### Tag Sampling:

The full tags file contains over one million rows, which is too large to load efficiently during every run. I decided to use a random sample of 500,000 rows instead. I chose random sampling over simply taking the first N rows, because taking the first rows would give deep coverage of only a handful of movies while leaving most of the dataset unrepresented. A random sample covers movies more evenly across the full collection.

##### Input Movie Exclusion:

A basic approach would skip the first result on the assumption that the input movie always ranks highest in similarity against itself. This assumption breaks down when tag coverage is uneven, as some movies end up with richer descriptions than others and may outscore the input movie. Instead, the system loops through the sorted results and skips the input movie wherever it appears, guaranteeing that it never appears in the recommendations regardless of its position.

##### Similarity Approach:

I built an initial version of the system using NearestNeighbors model with cosine metric. While it worked, I decided to replace it with direct cosine similarity, which provides equivalent results with more flexibility over the output. NearestNeighbors requires specifying the number of neighbors in advance and returns only those, while direct cosine similarity gives a complete list that can be sliced and filtered as needed.

## 5 - Limitations:

##### Tag Coverage:

About 36% of movies have no user tags at all. For these movies, the system relies entirely on genres to measure similarity, which as mentioned produces weaker recommendations. Older and more obscure films are the most affected, as they tend to have fewer tags or none at all.

##### Duplicate Titles:

As mentioned in the year extraction section, movies sharing a title can cause the system to recommend the wrong film. I noted this as a known limitation rather than a hard fix, since it affects a small number of movies. A future improved version will fix this issue letting the user decide which of the available options to choose from. 

##### Tag Sample Size:

Using a sample of 500,000 tags rather than the full dataset means some movies are better represented than others depending on which rows were included. A larger sample or the full dataset would improve recommendation quality, at the cost of higher memory usage and longer loading times.

##### Genre Weakness:

With only 19 unique genres, many movies share identical genre combinations. In the absence of tags, two completely different films can look identical to the system. This is the core motivation for including tags, but it remains a limitation for the portion of the collection that has no tags.


## 6 - Results: 

After testing the system with different movies  it was clear that genres alone wouldn't work. 

Once I added tags to the mix, the quality of the recommendations improved noticeably.

With genres alone the results were disappointing. Too many movies share the same genre combination, so the system had no real way to tell them apart and the recommendations felt random.

Tags make a real difference. A family adventure film now returns other films with a similar feel and tone, not just anything that happens to be tagged as adventure(which might be a Scifi, an animated movie for children, or a war movie). The recommendations start to make intuitive sense.

The system works best for popular films from the 1990s onwards, since those tend to have the most user tags. Older films and more obscure productions get weaker results, sometimes relying on genres alone, which as I mentioned is not ideal. But for a large part of the collection the results are useful.


## 7 - Future Improvements:

There are a few directions I would take this further given more time.

The most pressing improvement is handling duplicate titles properly. Right now the system automatically returns the first match, which may not be the film the user intended. Also, and a fuzzy title search that handles typos and partial matches would make the system more user friendly.

After that, adding KMeans clustering as a post-processing step would be a nice way to add variety to the recommendations. Instead of simply returning the five most similar films, I would cluster a larger set of candidates and pick one from each cluster, avoiding situations where all five results feel too similar to each other.

Loading the full tags dataset rather than a sample would improve coverage for less popular films.

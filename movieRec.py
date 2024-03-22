import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity


movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

############### Merging both datasets ####################
movies_merged = movies.merge(credits, on='title')

# Removing columns that won't be used in Recommendation
# select genres, id, keywords, title, overview, cast, crew

movies_merged = movies_merged[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
#print(movies_merged.shape)

################  Preprocessing  ##########################
#print(movies_merged.isnull().sum())
movies_merged.dropna(inplace=True)
#print(movies_merged.duplicated().sum())
#print(movies.iloc[0].genres)

# to convert string to integers for looping #
# fetching only name from this: [{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]
def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l
movies_merged['genres'] = movies_merged['genres'].apply(convert)
movies_merged['keywords'] = movies_merged['keywords'].apply(convert)

# for cast column, fetching only top 3
def convert2(obj):
    l = []
    cnt = 0
    for i in ast.literal_eval(obj):
        if cnt !=3:
            l.append(i['name'])
            cnt+=1
        else:
            break
    return l

movies_merged['cast'] = movies_merged['cast'].apply(convert2)
#print(movies_merged['cast'])

# for crew column, fetching director job only
def fetch_dir(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i['name'])
            break
    return l

movies_merged['crew'] = movies_merged['crew'].apply(fetch_dir)

# converting overview column (string to list)
movies_merged['overview'] = movies_merged['overview'].apply(lambda x:x.split())
#print(movies_merged['overview'].head())

# removing spaces b/w names to avoid confusion
movies_merged['genres'] = movies_merged['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies_merged['keywords'] = movies_merged['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies_merged['cast'] = movies_merged['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies_merged['crew'] = movies_merged['crew'].apply(lambda x:[i.replace(" ","") for i in x])
#print(movies_merged['crew'].head())
movies_merged['tags'] = movies_merged['overview'] + movies_merged['genres'] + movies_merged['keywords'] + movies_merged['cast'] + movies_merged['crew']
#print(movies_merged['tags'].head())
# new table
new_df = movies_merged[['movie_id', 'title', 'tags']]
#print(new_df.head())
# converting tags list into string
new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))
#print(new_df['tags'][0])

####################   text vectorization (Bag of words) for checking similarity (inverse of distance) between successive values of tags from 1 to 5000      ######################

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()
#print(vectors.shape)
#print(cv.get_feature_names_out())

# using stemming to remove similar names with prefix and suffix, then again convert to string
ps = PorterStemmer()
def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))

    return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)
#print(new_df['tags'][2])

# Now calculating cosine distance (not Euclidine) 4806 movies to 4806
similarity = cosine_similarity(vectors)
#print(similarity.shape) # (4806, 4806)
def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    # fetching values from 1 to 5 (not 0) which have lesser distance and sorting it w.r.t distance
    movies_list = sorted(list(enumerate(distances)), reverse=True, key= lambda x:x[1])[1:6]
    for i in movies_list:
        #print(i[0]) # will show index of top 5
        print(new_df.iloc[i[0]].title)
recommend('Small Soldiers')


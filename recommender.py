import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_title(index):
	return data[data.index == index]["title"].values[0]

def get_index(title):
	return data[data.title == title]["index"].values[0]

data = pd.read_csv("movie_dataset.csv")
# print(data)

features = ['keywords','cast','genres','director']

for feature in features:
	data[feature] = data[feature].fillna('')

# print(data.head())

def combine_features(row):
	try:
		return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]
	except:
		print ("Error:", row)

data["combined_features"] = data.apply(combine_features,axis=1)
print('--------------combined_features0----------------------')
print(data.head())
cv = CountVectorizer()

count_matrix = cv.fit_transform(data["combined_features"])
print(count_matrix)

cosine_sim = cosine_similarity(count_matrix)
movie_user_likes = "Superman Returns"

movie_index = get_index(movie_user_likes)

similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)

i=0
for element in sorted_similar_movies:
		print (get_title(element[0]))
		i=i+1
		if i>50:
			break

import pandas as pd
import numpy as np
from IPython.display import Image
import random
folds = 5
data = pd.read_csv("jester-data-1.csv")
#d = data.to_latex()
#text_file = open("Output-jester.txt", "w")
#text_file.write(d)
#text_file.close()
n_features = 2

user_ratings = data.values
print(user_ratings.shape[1])

print('Declaring arrays..')
user_preferences = np.random.random((user_ratings.shape[0], n_features))
joke_features = np.random.random((user_ratings.shape[1], n_features))

'''Change 10% to 99 and create 5 folds'''
print('Creating folds data with randomized null values (99)..')
foldsData = list()
for i in range(0,folds):
    user_ratingsTemp = user_ratings
    number_of_changed_values = int(len(user_ratings) / 10)
    for j in range(0,len(user_ratings)):
        randomJokeFeature = random.randint(0,user_ratings.shape[1]-1)
        '''Don't change the value if it was already 99. Decrease the counter then to know that a limited number of values have changed'''
        if user_ratingsTemp[j][randomJokeFeature] == 99:
            break
        user_ratingsTemp[j][randomJokeFeature] = 99
        number_of_changed_values = number_of_changed_values - 1
        if number_of_changed_values == 0:
            break
    foldsData.append(user_ratingsTemp)


def predict_rating(user_id, joke_id):
    """ Predict a rating given a user_id and an item_id."""
    user_preference = user_preferences[user_id]
    joke_preference = joke_features[joke_id]
    return user_preference.dot(joke_preference)

def train(user_id, joke_id, rating,user2, jokefeatures2, alpha=0.0001):
    # print item_id
    prediction_rating = predict_rating(user_id, joke_id)
    err = (prediction_rating - rating);
    # print err
    user_pref_values = user2[user_id][:]
    user2[user_id] -= alpha * err * jokefeatures2[joke_id]
    jokefeatures2[joke_id] -= alpha * err * user_pref_values
    return err


def sgd(iterations=100000):
    MSEs = []
    for i in range(0, folds):
        user_preferences = np.random.random((user_ratings.shape[0], n_features))
        joke_features = np.random.random((user_ratings.shape[1], n_features))
        for iteration in range(0, iterations):
            error = []
            for user_id in range(0, (user_preferences.shape[0])):
                for joke_id in range(0, joke_features.shape[0]):
                    rating = foldsData[i][user_id][joke_id]
                    if (rating != 99):
                        err = train(user_id, joke_id,rating,user_preferences, joke_features)
                        error.append(err)
            mse = (np.array(error) ** 2).mean()
            print(mse)
            MSEs.append(mse)

sgd()

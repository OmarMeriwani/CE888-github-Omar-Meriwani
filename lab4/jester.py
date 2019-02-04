import pandas as pd
import numpy as np
from IPython.display import Image
import random
np.set_printoptions(precision = 3)

#Read the data
data = pd.read_csv("jester-data-1.csv")
d = data.to_latex()
text_file = open("Output-jester.txt", "w")
text_file.write(d)
text_file.close()

user_ratings = data.values
user_preferences = range(0,user_ratings.shape[0])
joke_features = np.random.random((user_ratings.shape[1]))
#print(latent_joke_features)
#print(len(user_preferences))

user_preferences_array = list(range(len(user_preferences)))
user_preferences_array = [i for i in user_preferences]
joke_features_array = list(range(len(joke_features)))
joke_features_array = [i for i in joke_features]

count_of_randoms = int((len(user_preferences))/10)
print(count_of_randoms)
randoms_range = len(user_preferences) - 1
print(randoms_range)
user_features_randoms = [random.randint(0,randoms_range) for i in range(0,count_of_randoms)]
print(len(user_features_randoms))
changed_values = [user_preferences[i] for i in user_features_randoms]
for i in user_features_randoms:
    user_preferences_array[i] = 99


count_of_randoms = int((len(joke_features))/10)
print(count_of_randoms)
randoms_range = len(joke_features) - 1
print(randoms_range)
joke_features_randoms = [random.randint(0,randoms_range) for i in range(0,count_of_randoms)]
print(len(joke_features_randoms))
changed_values = [joke_features[i] for i in joke_features_randoms]
for i in joke_features_randoms:
    joke_features_array[i] = 99

print(joke_features_array)
'''
def predict_rating(user_id,item_id):
    user_preference = latent_user_preferences[user_id]
    item_preference = latent_joke_features[item_id]
    return user_preference.dot(item_preference)


def train(user_id, item_id, rating, alpha=0.0001):
    # print item_id
    prediction_rating = predict_rating(user_id, item_id)
    err = (prediction_rating - rating);
    # print err
    user_pref_values = latent_user_preferences[user_id][:]
    latent_user_preferences[user_id] -= alpha * err * latent_joke_features[item_id]
    latent_joke_features[item_id] -= alpha * err * user_pref_values
    return err
'''
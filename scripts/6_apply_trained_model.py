import cv2
import glob
import random
import numpy as np 
import os
from fnmatch import fnmatch
import re
import pandas as pd

fishface = cv2.face.FisherFaceRecognizer_create() #Initialize fisher face classifier

root = '../data/images/sorted_set'
pattern = "*.jpg"

dirs = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            dirs=dirs+[os.path.join(path, name)]

labels_df = pd.DataFrame(dirs, columns=['paths'])
labels_df['emotions'] = labels_df['paths']
labels_df['emotions'] = labels_df['emotions'].str.replace(r'.*sorted_set/', '')
labels_df['emotions'] = labels_df['emotions'].str.replace(r'/.*', '')

##### CREATE NUMBERED LABELS ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] #Emotion list
# 1 = angry/disgusted
# 1 = sad/worried/fear/surprise
# 2 = neutral/serious/contempt
# 3 = happy/other positive
labels_df['emotions'] = np.where(labels_df['emotions']=='neutral', '3', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='contempt', '3', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='fear', '2', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='sadness', '2', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='disgust', '1', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='anger', '1', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='surprise', '4', labels_df['emotions'])
labels_df['emotions'] = np.where(labels_df['emotions']=='happy', '4', labels_df['emotions'])

#### GET TAGGED IMAGES 
# tagged = pd.read_csv("../data/images/training_set/image_tags.csv", sep=',')
# labels_df = labels_df.append(tagged)
# labels_df = tagged
# labels_df = labels_df.sample(n=1000)

np.random.shuffle(labels_df.values)
training = labels_df

##### prediction data
root = '../data/images/cleaned_google_images'
pattern = "*.jpg"

dirs = []
for path, subdirs, files in os.walk(root):
    for name in files:
        if fnmatch(name, pattern):
            dirs=dirs+[os.path.join(path, name)]

dirs = [ x for x in dirs if not re.search('bad_images', x) ]
# dirs = dirs[:10]

def make_sets_final():
	training_data = []
	training_labels = pd.to_numeric(training['emotions']).tolist()
	prediction_data = []
	for item in training['paths']:
		image = cv2.imread(item) #open image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
		training_data.append(gray) #append image array to training data list

	for item in dirs: #repeat above process for prediction set
		image = cv2.imread(item)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		prediction_data.append(gray)

	return training_data, training_labels, prediction_data

def run_recognizer_final():
	prediction_labels=[]
	training_data, training_labels, prediction_data = make_sets_final()
	print("training fisher face classifier")
	print("size of training set is:", len(training), "images")
	fishface.train(training_data, np.asarray(training_labels))
	print("predicting classification set")
	cnt = 0
	correct = 0
	incorrect = 0
	for image in prediction_data:
		pred = fishface.predict(image)[0]
		prediction_labels.append(pred)
		print(str(cnt) + ' predicted emotion ' + str(fishface.predict(image)[0]))
	return prediction_labels


#Now run it
prediction_labels = run_recognizer_final()

predictions_df = pd.DataFrame(dirs, columns=['paths'])
predictions_df['emotions'] = prediction_labels
predictions_df['uidname'] = predictions_df['paths'].str.replace(r'.*cleaned_google_images/', '')
predictions_df['uidname'] = predictions_df['uidname'].str.replace(r'/.*', '')

print(predictions_df)

predictions_df.to_csv('../output/predicted_emotions.csv')




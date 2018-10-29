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

##### CREATE NUMBERED LABELS ["neutral", "anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"] 
# 1 = angry/disgusted
# 2 = sad/worried/fear/surprise
# 3 = neutral/serious/contempt
# 4 = happy/other positive
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


####

np.random.shuffle(labels_df.values)
training = labels_df[:int(len(labels_df)*0.8)] #get first 80% of file list
prediction = labels_df[-int(len(labels_df)*0.2):] #get last 20% of file list

def make_sets():
	training_data = []
	training_labels = pd.to_numeric(training['emotions']).tolist()
	prediction_data = []
	prediction_labels = pd.to_numeric(prediction['emotions']).tolist()
	for item in training['paths']:
		image = cv2.imread(item) #open image
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #convert to grayscale
		training_data.append(gray) #append image array to training data list

	for item in prediction['paths']: #repeat above process for prediction set
		image = cv2.imread(item)
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		prediction_data.append(gray)

	return training_data, training_labels, prediction_data, prediction_labels

def run_recognizer():
	training_data, training_labels, prediction_data, prediction_labels = make_sets()

	print("training fisher face classifier")
	print("size of training set is:", len(training), "images")
	fishface.train(training_data, np.asarray(training_labels))

	print("predicting classification set")
	cnt = 0
	correct = 0
	incorrect = 0
	for image in prediction_data:
		pred = fishface.predict(image)[0]
		print('true emotion ' + str(prediction_labels[cnt]))
		print('predicted emotion ' + str(fishface.predict(image)[0]))
		if pred == prediction_labels[cnt]:
			correct += 1
			cnt += 1
		else:
			incorrect += 1
			cnt += 1
	return ((100*correct)/(correct + incorrect))


#Now run it
metascore = []
# for i in range(0,1):
correct = run_recognizer()
print("got", correct, "percent correct!")
metascore.append(correct)

print("\n\nend score:", np.mean(metascore), "percent correct!")




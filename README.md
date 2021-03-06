# Description 

I measure the depicted emotions of political leaders around the world using their photos from google images and automatic emotion recognition techniques. I can then use this data to find results such as: 
- females leaders are more likely to be depicted as happy in photos
- left-wing leaders are more likely to be depicted as angry, and right-wing leaders are more likely to be depicted as neutral 
- the second major party (often the opposition) is slightly more likely to be depicted as negative than the first major party (often the government) 

(I have yet to test the statistical significance of these preliminary results...) 

![France leaders](https://github.com/hannahbull/emotions-politicians/blob/master/output/france_pols.png)

## Database of political parties and political orientations

I obtain a list of political parties from http://www.parlgov.org/: 
ParlGov database (Döring and Manow 2018)
Döring, Holger and Philip Manow. 2018. Parliaments and governments database (ParlGov): Information on parties, elections and cabinets in modern democracies. 

## Database of political leaders 

Scraped from searching the wikipedia page of the political parties in the ParlGov database

## Database of faces used to train the emotion recognition model

CK and CK+ database from http://www.consortium.ri.cmu.edu/ckagree/

- Kanade, T., Cohn, J. F., & Tian, Y. (2000). Comprehensive database for facial expression analysis. Proceedings of the Fourth IEEE International Conference on Automatic Face and Gesture Recognition (FG'00), Grenoble, France, 46-53.
- Lucey, P., Cohn, J. F., Kanade, T., Saragih, J., Ambadar, Z., & Matthews, I. (2010). The Extended Cohn-Kanade Dataset (CK+): A complete expression dataset for action unit and emotion-specified expression. Proceedings of the Third International Workshop on CVPR for Human Communicative Behavior Analysis (CVPR4HB 2010), San Francisco, USA, 94-101.

## Cleaning the face data

I use many existing python libraries and tutorials to extract the cropped faces, identify the correct person in each of the google images and train the model used to predict emotions

## Emotions

I do not use the 8 emotions in the CK database, but simplify to 4: 
1. angry = anger + disgust
2. sad = sad + fear
3. neutral = contempt + neutral
4. positive = happy + surprise

# Key results 

In the jupyter notebook 7_analysing_results.ipynb

# Requirements 

google-images-download from https://github.com/hardikvasa/google-images-download

google chrome chromedriver







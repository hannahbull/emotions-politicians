from google_images_download import google_images_download
import os
import pandas

cands = pandas.read_csv("../data/parties/list_of_candidates_final.csv", sep=',')

for i in range(0, len(cands.uidname)):
	personality = cands['leader'][i]
	print(personality)
	directory = cands['uidname'][i]
	#    print(directory)

	if not os.path.exists("../data/images/google_images/" + directory):
		os.makedirs("../data/images/google_images/" + directory, exist_ok=True)

	arguments = {"keywords": personality, "limit": 200, "print_urls": False,
		 "output_directory": "../data/images/google_images", "image_directory": directory, "chromedriver":"/usr/bin/chromedriver"}

	response = google_images_download.googleimagesdownload()
	absolute_image_paths = response.download(arguments)

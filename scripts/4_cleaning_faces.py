import face_recognition
import os
import pandas
import glob 
import numpy as np

politicians = [f for f in os.listdir("../data/images/google_images/")]

for politician in politicians:
	# politician='ramunas_karbauskis'
	print(politician)

	files = glob.glob("../data/images/cleaned_google_images/%s/*jpg" % politician)
	files = sorted(files) 
	files = sorted(files, key=len)

	number_good_images = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

	for i in range(0,9):

		try: 
			known_image = face_recognition.load_image_file(files[i])
			known_encoding = face_recognition.face_encodings(known_image)[0]

			j=0
			for fileno in files[:9]: 
				try:
					unknown_image = face_recognition.load_image_file(fileno)
					unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
					results = face_recognition.compare_faces([known_encoding], unknown_encoding)
					if results[0]: 
						j=j+1
				except: 
					print("fail")
			number_good_images[i] = j
			print(number_good_images)

		except:
			print("failed images " + politician)

	besti = np.argmax(number_good_images)
	print(besti)


	try: 
		known_image = face_recognition.load_image_file(files[besti])
		known_encoding = face_recognition.face_encodings(known_image)[0]
		if not os.path.exists("../data/images/cleaned_google_images/" + politician + "/bad_images"):
			os.makedirs("../data/images/cleaned_google_images/" + politician + "/bad_images", exist_ok=True)

		for fileno in files: 
			try:
				unknown_image = face_recognition.load_image_file(fileno)
				unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
				results = face_recognition.compare_faces([known_encoding], unknown_encoding)
				if not results[0]: 
					os.rename(fileno, "../data/images/cleaned_google_images/" + politician + "/bad_images/" + os.path.basename(fileno))
				print(fileno)
				print(results)
			except: 
				print("fail")
				os.rename(fileno, "../data/images/cleaned_google_images/" + politician + "/bad_images/" + os.path.basename(fileno))


	except:
		print("failed images " + politician)



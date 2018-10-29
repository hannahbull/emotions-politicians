import cv2
import os
import glob
import re

faceDet = cv2.CascadeClassifier("/home/hannah/anaconda3/share/OpenCV/haarcascade_frontalface_default.xml")

politicians = [f for f in os.listdir("../data/images/google_images/")]

for politician in politicians:
    # politician = "albert_rosti"

    # def detect_faces(politician):
    files = glob.glob("../data/images/google_images/%s/*" % politician)  # Get list of all images with emotion

    for f in files:
        try:
            if not os.path.exists("../data/images/cleaned_google_images/" + politician):
                os.makedirs("../data/images/cleaned_google_images/" + politician, exist_ok=True)

            filenumber = re.sub(".*/", "", f)
            filenumber = re.sub("\\..*", "", filenumber)

            frame = cv2.imread(f)  # Open image
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale

            face = faceDet.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10, minSize=(5, 5),
                                            flags=cv2.CASCADE_SCALE_IMAGE)
            facefeatures = face

            # Cut and save face
            subletter = 0
            for (x, y, w, h) in facefeatures:  # get coordinates and size of rectangle containing face
                subletter += 1
                print("face found in file: %s" % f)
                gray_new = gray[y:y + h, x:x + w]  # Cut the frame to size
                try:
                    out = cv2.resize(gray_new, (350, 350))  # Resize face so all images have same size
                    cv2.imwrite("../data/images/cleaned_google_images/%s/%s_%s_%s.jpg" % (politician, filenumber, subletter, politician), out)  # Write image
                    print("writing image")
                except:
                    pass  # If error, pass file
        except:
            pass

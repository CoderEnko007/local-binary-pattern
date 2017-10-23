from localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
from imutils import paths
import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--training", required=True)
ap.add_argument("-e", "--testing", required=True)
args = vars(ap.parse_args())

desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

for imagePath in paths.list_images(args["training"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)

    labels.append(imagePath.split("\\")[-2])
    data.append(hist)

model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)

for imagePath in paths.list_images(args["testing"]):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist)[0]

    cv2.putText(image, prediction, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.imshow("image", image)
    cv2.waitKey(0)
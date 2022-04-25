import cv2
import cv2 as cv
from deepface import DeepFace

cap = cv.VideoCapture(0)
while True:
    ret, frame = cap.read()


    predict = DeepFace.analyze(frame, enforce_detection=False)
    facecascade = cv2.CascadeClassifier(r"C:\Users\ishan\AppData\Local\Programs\Python\Python310\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml")
    faces = facecascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)


    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    font = cv.FONT_HERSHEY_SIMPLEX

    cv.putText(frame, predict['dominant_emotion'],
               (150, 150),
               font, 5,
               (0, 255, 255),
               3,
               cv.LINE_AA);

    cv.imshow("grey", frame)

    cv.waitKey(5)

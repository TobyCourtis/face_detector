import cv2
import sys

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
image_counter = 0
frame_number = 0

while (True):
    # Capture frame-by-frame - about 30 per second so set counter in here so takes photo
    # every 30 frames only
    #     img = cv2.flip(img, 1) ?? - mirror image if needed?
    frame_number += 1
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    k = cv2.waitKey(1)
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('FaceDetection', frame)


    if (k%256 == 27): #ESC Pressed - exit
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

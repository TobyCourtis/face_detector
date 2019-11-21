import cv2
import sys
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
image_counter = 0
frame_number = 0

print(sys.argv[1])
#print(os.path.isfile('../face_detector_data/toby/toby_face69.png'))
#if (not os.path.isfile(img_name)): #does not write over exitsing images
#check how many pics alreadu then image_counter = that number

while (True):
    # Capture frame-by-frame - about 30 per second so set counter in here so takes photo
    # every 30 frames only
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
        if(frame_number >= 10):
            crop = frame[y:y+h, x:x+w]
            img_name = '../face_detector_data/{}/{}_face{}.png'.format(sys.argv[1],sys.argv[1],image_counter)
            if(not os.path.exists(os.path.dirname(img_name))):
                try:
                    os.makedirs(os.path.dirname(img_name))
                except OSError: # Guard against race condition
                    print("error")
            cv2.imwrite(img_name, crop)
            print("Written: " + img_name)
            image_counter += 1
            frame_number = 0
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    # Display the resulting frame
    cv2.imshow('FaceDetection', frame)


    if (k%256 == 27): #ESC Pressed - exit
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

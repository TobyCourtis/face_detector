import cv2
import sys
import os

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)
frame_number = 0

print(sys.argv[1])
data_path = "../face_detector_data/{}".format(sys.argv[1])

if(os.path.exists(data_path)):
    image_counter = len([name for name in os.listdir(data_path) if os.path.isfile(os.path.join(data_path,name))])
else:
    image_counter = 0

while (True):
    # Capture frame-by-frame - about 30 per second so change - "if(frame_number >= 10):" to vary no photos taken
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

    for (x, y, w, h) in faces:
        if(frame_number >= 10):
            crop = frame[y:y+h, x:x+w]
            img_name = '../face_detector_data/{}/{}_face{}.png'.format(sys.argv[1],sys.argv[1],image_counter)
            if(not os.path.exists(os.path.dirname(img_name))):
                try:
                    os.makedirs(os.path.dirname(img_name))
                except OSError:
                    print("error")
            cv2.imwrite(img_name, crop)
            print("Written: " + img_name)
            image_counter += 1
            frame_number = 0
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('FaceDetection', frame)


    if (k%256 == 27): #ESC Pressed - exit
        break


# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

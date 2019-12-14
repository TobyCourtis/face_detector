import cv2

path = "../face_detector_data/toby/toby_face0.png"
image = cv2.imread(path, cv2.COLOR_BGR2GRAY)


scale = 25
while(True):
    #cv2.imshow("test", image)
    width = int(image.shape[1] * (scale/100))
    height = int(image.shape[0] * (scale/100))
    dim = (width, height)
    resized = cv2.resize(image, (32, 32), interpolation = cv2.INTER_AREA)
    cv2.imshow("resized_image", resized)
    k = cv2.waitKey(1)
    if (k%256 == 27): #ESC Pressed - exit
        break

cv2.destroyAllWindows()

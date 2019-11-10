import cv2

def show_webcam(mirror=False):
	img_counter = 0
	cam = cv2.VideoCapture(0)
	#cam.set(cv2.cv.CV_CAP_PROP_FPS, 10)
	cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
	cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
	cam.set(cv2.CAP_PROP_FPS, 30)
	while(True):
		ret_val, img = cam.read()
		if (mirror):
			img = cv2.flip(img, 1)
		# do stuff to img here
		# read in .sav at the start of the file and then classifier.classify(img)
		cv2.rectangle(img, (50,50), (200, 20), (0, 255, 0), 2)
		cv2.imshow('Webcam Input', img)
		if (cv2.waitKey(1) == 27):
			# esc to take photo
			print("writing image")
			img_name = "toby_face{}.png".format(img_counter)
			cv2.imwrite(img_name, img)
			print("{} written!".format(img_name))
			img_counter += 1
		elif (cv2.waitKey(1) == 32):
			print("breaking")
			break  # space to quit

	cv2.destroyAllWindows()


def main():
	show_webcam(mirror=True)


if __name__ == '__main__':
    main()

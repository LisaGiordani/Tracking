import numpy as np
import cv2
import matplotlib.pyplot as plt

roi_defined = False
verbose = True
treshold_voters = 100
 
def define_ROI(event, x, y, flags, param):
	global r,c,w,h,roi_defined
	# if the left mouse button was clicked, 
	# record the starting ROI coordinates 
	if event == cv2.EVENT_LBUTTONDOWN:
		r, c = x, y
		roi_defined = False
	# if the left mouse button was released,
	# record the ROI coordinates and dimensions
	elif event == cv2.EVENT_LBUTTONUP:
		r2, c2 = x, y
		h = abs(r2-r)
		w = abs(c2-c)
		r = min(r,r2)
		c = min(c,c2)  
		roi_defined = True
        
def get_gradient(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sobel derivatives
    frame_dx = cv2.Sobel(frame_gray, cv2.CV_64F, 1, 0)
    frame_dy = cv2.Sobel(frame_gray, cv2.CV_64F, 0, 1)

    # Gradient orientation
    orientation = np.arctan2(frame_dy, frame_dx)
    orientation = orientation*180/np.pi # from radian to degree
    orientation = np.round(orientation,-1) # round to the nearest ten
    for i in range(orientation.shape[0]):
        for j in range(orientation.shape[1]):
            if orientation[i,j] == -180:
                orientation[i,j] = 180
                
    # Gradient norm
    norm = np.hypot(frame_dx, frame_dy)    
    
    return orientation, norm


# Select a video
cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Ball.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Basket.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Car.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Sunshade.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Woman.mp4')

# take first frame of the video
ret,frame = cap.read()
# load the image, clone it, and setup the mouse callback function
clone = frame.copy()
cv2.namedWindow("First image")
cv2.setMouseCallback("First image", define_ROI)
 
# keep looping until the 'q' key is pressed
while True:
	# display the image and wait for a keypress
	cv2.imshow("First image", frame)
	key = cv2.waitKey(1) & 0xFF
	# if the ROI is defined, draw it!
	if (roi_defined):
		# draw a green rectangle around the region of interest
		cv2.rectangle(frame, (r,c), (r+h,c+w), (0, 255, 0), 2)
	# else reset the image...
	else:
		frame = clone.copy()
	# if the 'q' key is pressed, break from the loop
	if key == ord("q"):
		break


cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:
        #------------------------ Q3 ----------------------------------------#
        # Gradient
        orientation, norm = get_gradient(frame)
        # Display gradient
        orientation_gray = (orientation-np.min(orientation))/(np.max(orientation)-np.min(orientation))
        norm_gray = np.clip(norm, 0, 255)/255
        cv2.imshow('Orientation du gradient', orientation_gray)
        cv2.imshow('Norme du gradient', norm_gray)

        # Treshold defining voter pixels
        (non_voters_x, non_voters_y) = np.where(norm < treshold_voters)
        orientation_gray_red = np.zeros((orientation_gray.shape[0],orientation_gray.shape[1],3))
        orientation_gray_red[:,:,0] = orientation_gray
        orientation_gray_red[:,:,1] = orientation_gray
        orientation_gray_red[:,:,2] = orientation_gray
        for i in range(len(non_voters_x)):
            orientation_gray_red[non_voters_x[i],non_voters_y[i]] = [0, 0, 1]
        cv2.imshow('Orientation', orientation_gray_red)
        #--------------------------------------------------------------------#
        
        cv2.imshow('Sequence',frame)
        
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('Frame_%04d.png'%cpt,frame_tracked)
        cpt += 1
    else:
        break

cv2.destroyAllWindows()
cap.release()

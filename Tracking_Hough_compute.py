import numpy as np
import cv2
import matplotlib.pyplot as plt

roi_defined = False
verbose = True
treshold_voters = 100 # minimum treshold for a pixel to be voter
 
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


def get_RTable(roi_norm, roi_orientation):
    # Voter pixels
    (voters_x, voters_y) = np.where(roi_norm >= treshold_voters)
    nb_voters = len(voters_x)
    
    # Characteristics of the ROI
    width = roi_norm.shape[1]
    height = roi_norm.shape[0]
    roi_center = (height//2, width//2)

    RTable = {}
    for v in range(nb_voters):
        i = voters_x[v]
        j = voters_y[v]
        o = roi_orientation[i, j]
        if o in RTable :
            RTable[o].append((roi_center[0] - i, roi_center[1] - j))
        else: 
            RTable[o] = [(roi_center[0] - i, roi_center[1] - j)]
    print("RTable dim :", len(RTable))
    return RTable
    
    

def compute_Hough(frame, RTable, orientation, norm):
    # Initialisation
    H = np.zeros(frame.shape[0:2])
    
    # Characteristics of the frame
    width = frame.shape[1]
    height = frame.shape[0]
    
    # Voter pixels
    (voters_x, voters_y) = np.where(norm >= treshold_voters)
    nb_voters = len(voters_x)
    
    for v in range(nb_voters):
        i = voters_x[v]
        j = voters_y[v]
        o = orientation[i,j]
        if o in RTable:
            vectors = RTable[o]
            for (x,y) in vectors:
                if i+x >= 0 and i+x < height and j+y >= 0 and j+y < width: # in the frame
                    H[i+x,j+y] += 1 # vote
    
    print("Nb votes :", int(np.sum(H)))
    return H
    

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
 
track_window = (r,c,h,w)

# Setup the termination criteria: either 10 iterations,
# or move by less than epsilon pixels
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


#------------------------------- Q4 ------------------------------------------#
# Gradient
orientation, norm = get_gradient(frame)
roi_orientation = orientation[c:c+w, r:r+h]
roi_norm = norm[c:c+w, r:r+h]

# R-Table
RTable = get_RTable(roi_norm, roi_orientation)
#-----------------------------------------------------------------------------#

cpt = 1
while(1):
    ret ,frame = cap.read()
    if ret == True:
        #------------------------ Q3 ----------------------------------------#
        # Gradient
        orientation, norm = get_gradient(frame)
        
        #------------------------ Q4 ----------------------------------------#        
        # Hough transform
        H = compute_Hough(frame, RTable, orientation, norm)
        cv2.imshow('Hough transform',H/np.max(H))
        
        # Maximum of Hough transform
        maxH = np.max(H)
        maxH_x, maxH_y = np.where(H == maxH)
        maxH_x, maxH_y = int(np.mean(maxH_x)), int(np.mean(maxH_y[0])) # barycenter of maxima
        height, width = frame.shape[0:2]
        window_r = max(maxH_x-h//2,0)
        window_c = max(maxH_y-w//2,0)
        window_h = min(h,height-maxH_x-h//2)
        window_w = min(w,width-maxH_y-w//2)
        track_window = window_r, window_c, window_h, window_w
        ret = True
        #---------------------------------------------------------------------#
        
        # Draw a blue rectangle on the current image
        frame_tracked = cv2.rectangle(frame, (window_r,window_c), (window_r+window_h,window_c+window_w), (255,0,0) ,2)
        cv2.imshow('Sequence',frame_tracked)
        
        
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

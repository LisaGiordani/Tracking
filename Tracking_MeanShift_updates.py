import numpy as np
import cv2
import matplotlib.pyplot as plt

roi_defined = False
epsilon = 1 # termination criteria for meanshift algo : move by less than epsilon pixels
min_similarity = 20 # minimum similarity between the marginal histogram and the model histogram
idx_density = {"H": 0, "S": 1, "V": 2} # index of the densities in the HSV space
density = "H" # chosen density (H : Hue, S : saturation, V : value)
 
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

cap = cv2.VideoCapture('Sequences/Antoine_Mug.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Ball.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Basket.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Car.mp4')
#cap = cv2.VideoCapture('Sequences/VOT-Sunshade.mp4')
#ap = cv2.VideoCapture('Sequences/VOT-Woman.mp4')

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
# set up the ROI for tracking
roi = frame[c:c+w, r:r+h]
# conversion to Hue-Saturation-Value space
# 0 < H < 180 ; 0 < S < 255 ; 0 < V < 255
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# computation mask of the histogram:
# Pixels with S<30, V<20 or V>235 are ignored 
mask = cv2.inRange(hsv_roi, np.array((0.,30.,20.)), np.array((180.,255.,235.)))
# Marginal histogram of the Hue component
model_hist = cv2.calcHist([hsv_roi],[idx_density[density]],mask,[180],[0,180])
# Histogram values are normalised to [0,255]
cv2.normalize(model_hist,model_hist,0,255,cv2.NORM_MINMAX)
# Keep the first model histogram
model_hist_init = model_hist
# Display the histogram
plt.plot(model_hist)
plt.title("Model histogram")

# Setup the termination criteria: either 10 iterations,
# or move by less than epsilon pixels
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, epsilon)

cpt = 1
update_dst_ref = True
dst_ref = 0
while(1):
    ret, frame = cap.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
	    # Backproject the model histogram roi_hist onto the 
	    # current image hsv, i.e. dst(x,y) = roi_hist(hsv(0,x,y))
        dst = cv2.calcBackProject([hsv],[idx_density[density]],model_hist,[0,180],1)
        # Update the reference similarity
        if cpt == 1:
            dst_ref = np.mean(dst[c:c+w, r:r+h]) # or max(np.mean(dst[c:c+w, r:r+h]), dst_ref)
            dst_ref_init = dst_ref
            
        # ------------------------- Q2 : weights -----------------------------#
        # Apply a treshold to the brackprojection
        ret, dst_bin = cv2.threshold(dst, 127, 255, cv2.THRESH_BINARY)
        # Display the backprojection
        cv2.imshow("Binarized backprojection", dst_bin)

        # Apply meanshift to dst to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)
        (r,c,h,w) = track_window
        roi = frame[c:c+w, r:r+h]
        
        # ------------------------- Q1 ---------------------------------------#
        # Compute the marginal histogram
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0., 30., 20.)), np.array((180., 255., 235.)))
        roi_hist = cv2.calcHist([hsv_roi], [idx_density[density]], mask, [180], [0, 180])
        cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
        # Display the marginal histogram
        if cpt == 1:
            plt.figure()
            plt.title("Marginal histograms")
        plt.plot(roi_hist)
        
        # -------------------- Q2 : model histogram --------------------------#
        # Update of the model histogram
        print("mean(dst[c:c+w, r:r+h])", np.mean(dst[c:c+w, r:r+h]))
        if(np.mean(dst[c:c+w, r:r+h]) > np.floor(dst_ref) and cpt != 1): # high similarity
            model_hist = roi_hist
            dst_ref = np.mean(dst[c:c+w, r:r+h])
            print("Model histogram updated")
        elif(np.mean(dst[c:c+w, r:r+h]) < dst_ref_init/2): # very low similarity
            model_hist = model_hist_init
            dst_ref = dst_ref_init
            print("Model histogram re-initialized")
        
        # Draw a blue rectangle on the current image
        frame_tracked = cv2.rectangle(frame, (r,c), (r+h,c+w), (255,0,0) ,2)
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

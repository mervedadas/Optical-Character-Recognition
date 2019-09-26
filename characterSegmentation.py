import cv2
import numpy as np
img = cv2.imread('plate.jpg')

kernel = np.ones((3,3),np.uint8)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
ret,thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
# cv2.imshow("thresh",thresh)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
cv2.imshow("opening",opening)
copy= opening.copy()
contours, _ = cv2.findContours(copy, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #find contours
sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[3])

for i, ctr in enumerate(contours):
    # Get bounding box
    x, y, w, h = cv2.boundingRect(ctr)
    # Getting ROI
    roi = img[y:y+h, x:x+w]
    # print(roi.shape)
    # show ROI
    roi = cv2.resize(roi, (40, 40))
    cv2.imwrite(str(i)+'roi_imgs.png', roi)
    cv2.imshow('charachter'+str(i), roi)
    print(roi.shape)
    cv2.rectangle(copy,(x,y),( x + w, y + h ),(0,0,255),1)

cv2.imshow('marked areas',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

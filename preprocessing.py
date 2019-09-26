import cv2
import numpy as np
import os
from autoCanny import auto_canny

trainset=[]
size=(40,40)
kernel =np.ones((3,3),np.uint8)
for i in range(0,10):
    for j in range(1,3795):
            path ="trainingSet/"+str(i)+"/"+"img ("+str(j)+").jpg"
            img=cv2.imread(path,0)
            img=cv2.resize(img,size)
            blur = cv2.GaussianBlur(img, (3,3), 0)
            ret, binary= cv2.threshold(blur, 127, 255, cv2.THRESH_BINARY)
            canny = auto_canny(binary)
            cv2.imshow(str(i)+"image"+str(j),canny)
            trainset.append(canny)
            # cv2.imwrite("dataset/"+str(i)+"/"+"img ("+str(j)+").jpg",canny)


cv2.waitKey()
cv2.destroyAllWindows()




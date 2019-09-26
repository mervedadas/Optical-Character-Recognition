import cv2
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img = cv2.imread('plate.jpg',0)
cv2.imshow('image',img)

text = pytesseract.image_to_string(img, lang = 'eng')
print(text)

cv2.waitKey(0)
cv2.destroyAllWindows()
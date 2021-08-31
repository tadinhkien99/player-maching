import cv2
import numpy as np

src=cv2.imread('test.jpg')


cv2.circle(src,(94,219), 2, (0,190,255),-1)
cv2.circle(src,(96,201), 2, (0,190,255),-1)
cv2.circle(src,(104,188), 2, (0,190,255),-1)
cv2.circle(src,(117,184), 2, (0,190,255),-1)

cv2.circle(src,(132,184), 2, (0,190,255),-1)
cv2.circle(src,(145,186), 2, (0,190,255),-1)
cv2.circle(src,(156,196), 2, (0,190,255),-1)
cv2.circle(src,(98,207), 2, (0,190,255),-1)

cv2.circle(src,(106,194), 2, (0,190,255),-1)
cv2.circle(src,(120,193), 2, (0,190,255),-1)
cv2.circle(src,(135,193), 2, (0,190,255),-1)
cv2.circle(src,(147,195), 2, (0,190,255),-1)


#nose
cv2.circle(src,(184,211), 2, (0,190,255),-1)
cv2.circle(src,(190,233), 2, (0,190,255),-1)
cv2.circle(src,(195,255), 2, (0,190,255),-1)
cv2.circle(src,(201,277), 2, (0,190,255),-1)

cv2.circle(src,(209,290), 2, (0,190,255),-1)
cv2.circle(src,(195,290), 2, (0,190,255),-1)
cv2.circle(src,(174,290), 2, (0,190,255),-1)
cv2.circle(src,(170,278), 2, (0,190,255),-1)

cv2.circle(src,(220,279), 2, (0,190,255),-1)
cv2.circle(src,(237,268), 2, (0,190,255),-1)
cv2.circle(src,(227,258), 2, (0,190,255),-1)

cv2.circle(src,(243,379), 5, (0,190,255),-1)
cv2.circle(src,(145,88), 5, (0,190,255),-1)



cv2.imshow("src", src)
cv2.waitKey(0)
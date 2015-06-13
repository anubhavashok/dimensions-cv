import cv2
import numpy as np

fc_img = cv2.imread('coins.jpg', 1)


def fc(img, circles):
    sift = cv2.SIFT()
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   
    
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    kp1, des1 = sift.detectAndCompute(fc_img, None)
    max = None
    for circle in circles[0,:]:
        start = circle[:2] - circle[2]
        end = circle[:2] + circle[2]
        cutout = img[start[1]:end[1], start[0]:end[0]]
        cv2.imshow('asd', cutout)
        cv2.waitKey()
        cv2.destroyAllWindows()
        kp2, des2 = sift.detectAndCompute(cutout, None)
        matches = flann.knnMatch(des1, des2, k=2)
        n = len(matches)
        if (max is None) or (n > max[1] + 10):
            max = (circle, n) 
        elif abs(max[1] - n) <= 10:
            max = (circle, n) if circle[2] > max[0][2] else max
    return max[0]

def findCircles(himg):
    circles = None
    val = 1.2
    delta = 0.3
    while circles is None:
        circles = cv2.HoughCircles(himg, cv2.cv.CV_HOUGH_GRADIENT, val, 20)
        val += delta
    circles = np.uint16(np.around(circles))
    return circles

def floodImage(img, sample_spot):
    h, w = img.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    mask[:] = 0
    flooded = img.copy()
    cv2.floodFill(flooded, mask, tuple(sample_spot), (255, 255, 255), (4,)*3, (10,)*3, 4)
    return flooded

# radius of coin in cm
coin_diameter = 2.3 # cm
pixels_per_cm = None

img = cv2.imread('coins_query.jpg', 1)
himg = cv2.imread('coins_query.jpg', 0)
cimg = cv2.cvtColor(himg, cv2.COLOR_GRAY2BGR)


# find coin
circles = findCircles(himg)

# draw circles
for i in circles[0, :]:
    # draw the outer circle
    cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

# get corresponding coin
circle = fc(img, circles)

pixels_per_cm = (circle[2]*2)/coin_diameter

# flood image
sample_spot = circle[:2]+circle[2]+(10)
flooded = floodImage(img, sample_spot)
cv2.imshow('a', flooded)

# detect object (find largest contour)
t = cv2.cvtColor(flooded, cv2.COLOR_BGR2GRAY)
ret, nmask = cv2.threshold(t, 230, 255, cv2.THRESH_BINARY)
cnts, _ = cv2.findContours(nmask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

x, y, w, h = cv2.boundingRect(cnts[1])
pt1 = (x, y)
pt2 = (x+w, y+h)
cv2.rectangle(img, pt1, pt2, (100, 100, 100))
w_cm = round(w / pixels_per_cm, 2)
h_cm = round(h / pixels_per_cm, 2)
print w_cm, h_cm
h_text = str(h_cm) + " cm"
w_text = str(w_cm) + " cm"
# write w, h as text in middle of bounding line
w_y = y+h/2
w_x = x - cv2.getTextSize(h_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
cv2.putText(img, h_text, (w_x, w_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)

h_y = y + h + 5 + cv2.getTextSize(w_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][1]
h_x = x + w/2 - cv2.getTextSize(w_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]/2 + 1
cv2.putText(img, w_text, (h_x, h_y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, thickness=2)


cv2.circle(cimg, (circle[0], circle[1]), circle[2], (255, 0, 0), 2)

cv2.imshow('detected circles', img)
cv2.waitKey()
cv2.destroyAllWindows()


# background color
# floodfill
# remove blobs
# cvt to binary image
# remove blobs under a certain radius


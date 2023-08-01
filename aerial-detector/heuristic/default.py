import cv2
import numpy as np
max_distance = 50  # Adjust this value as per your requirement
def find_closest_contours(contours):
    clustered_contours = []
    for contour in contours:
        x, y, _, _ = cv2.boundingRect(contour)
        cluster_found = False
        
        for cluster in clustered_contours:
            cx, cy, _, _ = cv2.boundingRect(cluster[0])
            distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            
            if distance < max_distance:
                cluster.append(contour)
                cluster_found = True
                break
        
        if not cluster_found:
            clustered_contours.append([contour])
    
    return clustered_contours

def contour_ammalgomater(contours, mask):
    # Cluster nearby contours
    
    clustered_contours = find_closest_contours(contours)

    # Now you have clusters of nearby contours
    # You can loop through the clustered_contours and apply operations on each cluster
    for cluster in clustered_contours:
        # Do something with the cluster
        # For example, calculate the bounding rectangle around the cluster
        x, y, w, h = cv2.boundingRect(cv2.drawContours(np.zeros_like(mask), cluster, -1, 255, thickness=cv2.FILLED))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)



input_img = cv2.imread("C:\\Users\\EaglesonLabs\\object-detection\\aerial-detector\\Aero Landing Zone Object Detection.v1i.yolov8\\train\\images\\75degree_1_MP4-12_jpg.rf.9d065395d94692eaa9f35df62492de4a.jpg")
print(input_img.shape)
img = cv2.resize(input_img, (1500, 1200))
# Make a copy to draw contour outline
input_image_cpy = img.copy()
 
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
 
# define range of red color in HSV
lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])
 
# define range of green color in HSV
# lower_green = np.array([40, 20, 50])
# upper_green = np.array([90, 255, 255])
 
# define range of blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([130, 255, 255])
 
# create a mask for red color
mask_red = cv2.inRange(hsv, lower_red, upper_red)
 
# find contours in the red mask
contours_red, _ = cv2.findContours(mask_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_ammalgomater(contours_red, mask_red)

# create a mask for green color
#mask_green = cv2.inRange(hsv, lower_green, upper_green) 
 
# find contours in the green mask
#contours_green, _ = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# create a mask for blue color
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# find contours in the blue mask
contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contour_ammalgomater(contours_blue, mask_blue)

# # loop through the red contours and draw a rectangle around them
# for cnt in contours_red:
#     contour_area = cv2.contourArea(cnt)
#     if contour_area > 10:
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
#         cv2.putText(img, 'Red', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
     
# # loop through the green contours and draw a rectangle around them
# # for cnt in contours_green:
# #     contour_area = cv2.contourArea(cnt)
# #     if contour_area > 1000:
# #         x, y, w, h = cv2.boundingRect(cnt)
# #         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# #         cv2.putText(img, 'Green', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
 
# # loop through the blue contours and draw a rectangle around them
# for cnt in contours_blue:
#     contour_area = cv2.contourArea(cnt)
#     if contour_area > 10:
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         cv2.putText(img, 'Blue', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
 
# Display final output for multiple color detection opencv python
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2, os, imutils
import numpy as np

max_distance = 100  # Adjust this value as per your requirement

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

def contour_ammalgomater(img, contours, mask):
    # Cluster nearby contours
    
    clustered_contours = find_closest_contours(contours)

    # Now you have clusters of nearby contours
    # You can loop through the clustered_contours and apply operations on each cluster
    for cluster in clustered_contours:
        # Do something with the cluster
        # For example, calculate the bounding rectangle around the cluster
        x, y, w, h = cv2.boundingRect(cv2.drawContours(np.zeros_like(mask), cluster, -1, 255, thickness=cv2.FILLED))
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 1)


params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 1
params.maxThreshold = 500

# Filter by Area.
params.filterByArea = True
params.minArea = 300

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.8

# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.2

# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


image_folders = [
    'C:\\Users\\EaglesonLabs\\object-detection\\aerial-detector\\Aero Landing Zone Object Detection.v2i.yolov8\\test\\images',
    'C:\\Users\\EaglesonLabs\\object-detection\\aerial-detector\\Aero Landing Zone Object Detection.v2i.yolov8\\train\\images',
    'C:\\Users\\EaglesonLabs\\object-detection\\aerial-detector\\Aero Landing Zone Object Detection.v2i.yolov8\\valid\\images'
                 ]
for image_folder in image_folders:
    print(image_folder)
    images = os.listdir(image_folder)
    for image in images:
        if image is not None:
            image = os.path.join(image_folder, image)
            input_img = cv2.imread(image)
            img = cv2.resize(input_img, (1500, 1200))
            # Make a copy to draw contour outline
            img = img.copy()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # define range of red color in HSV
            lower_red = np.array([0, 50, 50])
            upper_red = np.array([10, 255, 255])   
            

            # define range of blue color in HSV
            lower_blue = np.array([100, 50, 50])
            upper_blue = np.array([130, 255, 255])
            blurred = cv2.GaussianBlur(img,(5,5),cv2.BORDER_DEFAULT)
            blurred = cv2.GaussianBlur(blurred,(5,5),cv2.BORDER_DEFAULT)


            ret, thresh2 = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY_INV)
            keypoints=[]
            ret, thresh2 = cv2.threshold(thresh2, 249, 255, cv2.THRESH_BINARY_INV)
            ret, thresh2 = cv2.threshold(thresh2, 1, 255, cv2.THRESH_BINARY_INV)
            ret, thresh2 = cv2.threshold(thresh2, 249, 255, cv2.THRESH_BINARY_INV)
            ret, thresh2 = cv2.threshold(thresh2, 1, 255, cv2.THRESH_BINARY_INV)
            keypoints.append(detector.detect(thresh2))

            
            hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)         
            # create a mask for red color
            mask_red = cv2.inRange(hsv, lower_red, upper_red)
            # find contours in the red mask
            contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # create a mask for blue color
            mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
            # find contours in the blue mask
            contours_blue, _ = cv2.findContours(mask_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            contour_ammalgomater(img, contours_blue, mask_blue)
            contour_ammalgomater(img, contours_red, mask_red)
            
            points = 0
            for keypoint in keypoints:
                # Draw blobs on our image as red circles
                blank = np.zeros((4, 4)) 
                img = cv2.drawKeypoints(img, keypoint, blank, (0, 0, 0),
                                        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                points += len(keypoint)
            text = "Number of Circular Blobs: " + str(points)
            cv2.putText(img, text, (20, 550),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
            
            # Display final output for multiple color detection opencv python
            cv2.imshow('image', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
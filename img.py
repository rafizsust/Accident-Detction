import cv2    
import time

# Initialize variables
cpt = 0
maxFrames = 130  # Maximum number of frames to extract
count = 0

# Open video file
cap = cv2.VideoCapture('cctv.mp4')

while cpt < maxFrames:
    # Read a frame from the video
    ret, frame = cap.read()
    
    # Check if frame was successfully read
    if not ret:
        break
    
    count += 1
    
    # Extract every third frame
    if count % 3 != 0:
        continue
    
    # Resize frame
    frame = cv2.resize(frame, (1080, 500))
    
    # Display the frame
    cv2.imshow("Frame", frame)
    
    # Save the frame as an image
    cv2.imwrite(r"C\rafiz\directory..\images\cctv_road_accident_%d.jpg" % cpt, frame)
    
    # Pause briefly to slow down the process
    time.sleep(0.01)
    
    cpt += 1
    
    # Check for ESC key press to exit
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release video capture object and close all windows
cap.release()   
cv2.destroyAllWindows()


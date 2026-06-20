import cv2 as cv

# Read video
print("reading the video file...")
cap = cv.VideoCapture('data/PSG_vs_Bayern_Munchen.mp4')

if not cap.isOpened():
    print("Error: Cannot open video file")
    exit()

# Get video properties
fps = cap.get(cv.CAP_PROP_FPS)
width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

print(f"FPS: {fps}, Size: {width}x{height}, Frames: {frame_count}")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv.imshow('Video', frame)

    # Wait time to match video FPS (ESC to exit)
    if cv.waitKey(int(1000/fps)) & 0xFF == 27:
        break

cap.release()
cv.destroyAllWindows()
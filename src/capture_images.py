from pypylon import pylon
import cv2
import time

# Constants
CONVEYOR_SPEED = 5  # meters per minute
FRAME_WIDTH = 1000  # adjust based on your camera and needs
FRAMES_PER_METER = 10  # adjust based on desired resolution

# Calculate frame rate
frames_per_minute = CONVEYOR_SPEED * FRAMES_PER_METER
frame_interval = 60.0 / frames_per_minute  # in seconds

# Initialize the camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()

# Set camera parameters (adjust as needed)
camera.Width.SetValue(FRAME_WIDTH)
camera.Height.SetValue(FRAME_WIDTH)  # Assuming a square frame
camera.PixelFormat.SetValue('Mono8')  # or 'RGB8' for color
camera.ExposureTime.SetValue(10000)  # in microseconds

# Start grabbing images
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

frame_count = 0
start_time = time.time()

try:
    while camera.IsGrabbing():
        if time.time() - start_time >= frame_interval:
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            
            if grabResult.GrabSucceeded():
                # Convert to OpenCV image
                image = grabResult.Array
                
                # Save or process the image
                cv2.imwrite(f'egg_image_{frame_count}.png', image)
                
                frame_count += 1
                print(f"Captured frame {frame_count}")
            
            grabResult.Release()
            start_time = time.time()  # Reset the timer

except KeyboardInterrupt:
    print("Stopping image capture.")

finally:
    camera.StopGrabbing()
    camera.Close()

print(f"Total frames captured: {frame_count}")
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import label
from skimage.color import label2rgb
import cv2
from mpl_toolkits.mplot3d import Axes3D
from EyeTracking.EyeTracker.OrloskyPupilDetector import crop_to_aspect_ratio, get_darkest_area, apply_binary_threshold, mask_outside_square, process_frames
import os
import time
"""
print(cv2.__version__)
# Replace 'default_image.jpg' with your actual default image filename
image_path = 'Images\default_image.jpg'

# Open the image
img = Image.open(image_path)

# Show the image (optional)
#img.show()


import matplotlib.pyplot as plt

# Convert PIL image to numpy array (RGB)
arr = np.array(img)

# Convert to grayscale using OpenCV
gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

# Perform Gaussian blurs and subsample
blurred1 = cv2.GaussianBlur(gray, (0, 0), sigmaX=2)
subsampled1 = blurred1[::2, ::2]

blurred2 = cv2.GaussianBlur(subsampled1, (0, 0), sigmaX=2)
subsampled2 = blurred2[::2, ::2]

# Threshold to create binary image
threshold = np.mean(subsampled2)
_, binary = cv2.threshold(subsampled2, threshold, 255, cv2.THRESH_BINARY)

# Label blobs using connected components
num_labels, labels = cv2.connectedComponents(binary.astype(np.uint8))

# Create a color label image for visualization
label_hue = np.uint8(179 * labels / np.max(labels))
blank_ch = 255 * np.ones_like(label_hue)
image_label_overlay = cv2.merge([label_hue, blank_ch, blank_ch])
image_label_overlay = cv2.cvtColor(image_label_overlay, cv2.COLOR_HSV2RGB)
image_label_overlay[label_hue == 0] = 0

# Display results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[1].imshow(subsampled2, cmap='gray')
axes[1].set_title('Blurred & Subsampled')
axes[2].imshow(image_label_overlay)
axes[2].set_title('Identified Blobs')
for ax in axes:
    ax.axis('off')
plt.tight_layout()
plt.show()

# Set up the SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 30
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs in the grayscale image
keypoints = detector.detect(gray)

# Draw detected blobs as red circles.
img_with_keypoints = cv2.drawKeypoints(
    arr, keypoints, np.array([]), (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

plt.figure(figsize=(6, 6))
plt.imshow(img_with_keypoints)
plt.title('Blobs Detected')
plt.axis('off')
# Ring detector using HoughCircles

# Convert to grayscale if not already
gray_for_circles = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

# Apply a median blur to reduce noise and improve circle detection
blurred = cv2.medianBlur(gray_for_circles, 5)

# Detect circles using HoughCircles
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=100
)

# Draw detected rings (circles)
output = arr.copy()
# Make HoughCircles less sensitive by increasing param2 (accumulator threshold)
circles = cv2.HoughCircles(
    blurred,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=20,
    param1=60,
    param2=60,  # Increased from 30 to 50 for less sensitivity
    minRadius=10,
    maxRadius=100
)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # Draw the outer circle
        # To make detection even less sensitive, increase param2 further if needed
        cv2.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)

plt.show()
plt.figure(figsize=(6, 6))
plt.imshow(output)
plt.title('Detected Rings')
plt.axis('off')
plt.show()

# Source - https://stackoverflow.com/q
# Posted by Anidh Singh
# Retrieved 2025-11-14, License - CC BY-SA 4.0

image = cv2.imread(image_path)
image_copy_new=cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
retval, thresholded = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY_INV)
plt.imshow(thresholded,cmap="gray")
plt.show()
contours, hierarchy = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
image_copy = image.copy()  # create a copy to draw on
for cnt in contours:
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
    (x, y, w, h) = cv2.boundingRect(cnt)
    ar = w / float(h)
    if w*h > 20 and 0.9 < ar < 1.1:  # filtering condition
        cv2.drawContours(image_copy, [cnt], 0, (0, 255, 0), 2)  # draw in green

plt.figure(figsize=(6, 6))
plt.imshow(cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB))
plt.title('Annotated Image')
plt.axis('off')
plt.show()

# Keep the plot window open until closed by the user
plt.show()

print("Done")


# Detect circular structure outlines (not blobs) of different sizes using HoughCircles

# Convert to grayscale if not already
gray_outline = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

# Apply a median blur to reduce noise
blurred_outline = cv2.medianBlur(gray_outline, 5)

# Detect circles using HoughCircles (tune parameters for your use case)
# Detect circles using HoughCircles (tune parameters for your use case)
circles_outline = circles

# Post-process: keep only circles that separate regions of distinct intensity
def is_distinct_intensity(circle, image, threshold=0.2):
    x, y, r = int(circle[0]), int(circle[1]), int(circle[2])
    # Create masks for inside and outside the circle
    Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
    dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
    mask_inner = dist_from_center < (r - 2)
    mask_outer = (dist_from_center > (r + 2)) & (dist_from_center < (r + 8))
    # Avoid out-of-bounds
    if np.sum(mask_inner) < 10 or np.sum(mask_outer) < 10:
        return False
    mean_inner = np.mean(image[mask_inner])
    mean_outer = np.mean(image[mask_outer])
    # Normalize difference
    diff = abs(mean_inner - mean_outer) / 255.0
    return diff > threshold

if circles_outline is not None:
    filtered_circles = []
    for c in circles_outline[0, :]:
        if is_distinct_intensity(c, gray_outline):
            filtered_circles.append(c)
    if filtered_circles:
        circles_outline = np.array([filtered_circles])
    else:
        circles_outline = None

# Draw detected circle outlines
output_outline = arr.copy()
if circles_outline is not None:
    circles_outline = np.uint16(np.around(circles_outline))
    for i in circles_outline[0, :]:
        # Draw only the circle outline (no center dot)
        cv2.circle(output_outline, (i[0], i[1]), i[2], (255, 0, 0), 2)

plt.figure(figsize=(6, 6))
plt.imshow(output_outline)
plt.title('Detected Circular Structure Outlines')
plt.axis('off')
plt.show()

# Prepare grid for 3D plot
X = np.arange(gray.shape[1])
Y = np.arange(gray.shape[0])
X, Y = np.meshgrid(X, Y)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, gray, cmap='gray', linewidth=0, antialiased=False)
ax.set_title('Grayscale Image as 3D Surface')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Intensity')
plt.tight_layout()
ax.plot_surface(X, Y, blurred1, cmap='gray', linewidth=0, antialiased=False)
# If circles were detected, focus the 3D plot around the largest detected circle
if circles_outline is not None:
    # Find the largest circle by radius
    largest_circle = max(circles_outline[0, :], key=lambda c: c[2])
    x0, y0, r = largest_circle
    # Define a window around the circle (with some margin)
    margin = int(r * 1.2)
    x_start = max(0, x0 - margin)
    x_end = min(gray.shape[1], x0 + margin)
    y_start = max(0, y0 - margin)
    y_end = min(gray.shape[0], y0 + margin)
    # Crop the region of interest
    X_crop = X[y_start:y_end, x_start:x_end]
    Y_crop = Y[y_start:y_end, x_start:x_end]
    blurred1_crop = blurred1[y_start:y_end, x_start:x_end]
    ax.clear()
    ax.plot_surface(X_crop, Y_crop, blurred1_crop, cmap='gray', linewidth=0, antialiased=False)
    ax.set_title('3D Surface Around Largest Detected Circle')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Intensity')
plt.show()
if circles is not None:
    # Show the same region (around the largest detected circle) on the original image
    largest_circle = max(circles[0, :], key=lambda c: c[2])
    x0, y0, r = largest_circle
    margin = int(r * 1.2)
    x_start = max(0, x0 - margin)
    x_end = min(arr.shape[1], x0 + margin)
    y_start = max(0, y0 - margin)
    y_end = min(arr.shape[0], y0 + margin)
    region = arr[y_start:y_end, x_start:x_end]

    plt.figure(figsize=(6, 6))
    plt.imshow(region)
    plt.title('Region Around Largest Detected Circle')
    plt.axis('off')
    plt.show()
    print(f"Largest detected circle center: (x={x0}, y={y0}), radius={r}")
    # Plot the original image with an arrow from the image center to the largest circle center
    plt.figure(figsize=(6, 6))
    plt.imshow(arr)
    plt.title('Arrow: Image Center to Largest Circle Center')
    plt.axis('off')"""

if __name__ == "__main__":
    print(cv2.__version__)
    images_folder = 'Images'
    image_files = [f for f in os.listdir(images_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]

    for image_file in image_files:
        image_path = r'Images\\' + image_file

        #image_path = r'Images\default_image2.jpg'
        cv2.destroyAllWindows()
        # Open the image
        img = cv2.imread(image_path)
        cv2.imshow('Input Image', img)
        cv2.waitKey(1)  # Add this line to ensure the window is displayed
        
        frame = img.copy()
        debug_mode_on = True
        # Crop and resize frame
        #input("Press Enter to process this image...")
        frame = crop_to_aspect_ratio(frame, frame.shape[1], frame.shape[0])

        #find the darkest point
        darkest_point = get_darkest_area(frame)

        """if debug_mode_on:
            darkest_image = frame.copy()
            cv2.circle(darkest_image, darkest_point, 10, (0, 0, 255), -1)
            cv2.imshow('Darkest image patch', darkest_image)"""

        # Convert to grayscale to handle pixel value operations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

        # apply thresholding operations at different levels
        # at least one should give us a good ellipse segment
        thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
        thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

        thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
        thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)

        thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
        thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)

        #take the three images thresholded at different levels and process them
        pupil_rotated_rect, centre_x, centre_y = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, True)
        # Calculate px_per_mm assuming the pupil's smallest diameter is 2mm
        # pupil_rotated_rect[1] gives (width, height); take the smaller as the pupil diameter in px
        pupil_diameter_px = min(pupil_rotated_rect[1])
        px_per_mm = pupil_diameter_px / 4.0
        print(f"Pixels per mm: {px_per_mm:.2f} px/mm")
        # Calculate the vector from the image center to (centre_x, centre_y)
        image_center = (frame.shape[1] // 2, frame.shape[0] // 2)
        vector = (centre_x - image_center[0], centre_y - image_center[1])
        print(f"Vector from image center to pupil center: {vector}")
        # Draw the vector from image center to pupil center
        output_img = frame.copy()
        #cv2.destroyAllWindows()
        # Draw the detected ellipse (pupil) on the output image
        cv2.ellipse(output_img, pupil_rotated_rect, (255, 255, 0), 2)
        # Draw the arrowed line from image center to pupil center
        cv2.arrowedLine(
            output_img,
            image_center,
            (int(centre_x), int(centre_y)),
            (0, 0, 255),
            3,
            tipLength=0.2
        )
        # Draw the pupil center and image center for clarity
        cv2.circle(output_img, (int(centre_x), int(centre_y)), 6, (0, 255, 0), -1)
        cv2.circle(output_img, image_center, 6, (255, 0, 0), -1)

        cv2.imshow("Vector from Image Center to Pupil Center", output_img)
        cv2.waitKey(1)  # Add this line to ensure the window is displayed
        # Calculate the physical length of the arrowed line in mm
        arrow_length_px = np.linalg.norm(np.array(vector))
        arrow_length_px = np.linalg.norm(np.array(vector))
        arrow_length_mm = arrow_length_px / px_per_mm
        print(f"Physical length of the arrowed line: {arrow_length_mm:.2f} mm")

        # Normalize the arrowed line (vector) components to range 0-255
        vector = np.array((int(centre_x), int(centre_y))) - np.array(image_center)
        norm_x = int(255 * abs(vector[0]) / (px_per_mm*4))
        norm_y = int(255 * abs(vector[1]) / (px_per_mm*4))
        norm_x = min(norm_x, 255)
        norm_y = min(norm_y, 255)

        # Determine direction
        x_dir = "right" if vector[0] > 0 else "left"
        y_dir = "down" if vector[1] > 0 else "up"

        print(f"X component: {norm_x} ({x_dir})")
        print(f"Y component: {norm_y} ({y_dir})")


        input("Press Enter to continue to the next image...")
        

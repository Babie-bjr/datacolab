# datacolab
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from google.colab import drive

# Function to convert pixels to cm
def pixel_to_cm(pixels, dpi=300):
    inches = pixels / dpi
    return inches * 2.54

# Function to classify size based on diameter
def get_size_class(diameter):
    if diameter > 6.2:
        return 1
    elif 5.9 <= diameter <= 6.2:
        return 2
    elif 5.3 <= diameter <= 5.8:
        return 3
    elif 4.6 <= diameter <= 5.2:
        return 4
    elif 3.8 <= diameter <= 4.5:
        return 5
    else:
        return 0

# Mount Google Drive
drive.mount('/content/drive')

# Define folder paths
input_folder = '/content/drive/MyDrive/Project/Dataset'  # Path to your folder containing images
output_folder = '/content/drive/MyDrive/Project/PhotoDataset'  # Path to the folder where processed images will be saved

# Check if output folder exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Create a list to store the data for the Excel file
data = []

# Loop over all images in the input folder
for idx, filename in enumerate(os.listdir(input_folder)):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add other image formats if necessary
        print(f"Processing {filename}...")

        # Read the image
        original_image = cv2.imread(os.path.join(input_folder, filename))

        if original_image is None:
            print(f"Error: Could not load image from {filename}.")
            continue

        gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, 128, 255, cv2.THRESH_BINARY_INV)
        kernel = np.ones((5, 5), np.uint8)
        binary_image_cleaned = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
        contours1, _ = cv2.findContours(binary_image_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours_image = original_image.copy()

        if contours1:
            largest_contour1 = max(contours1, key=cv2.contourArea)
            cv2.drawContours(contours_image, [largest_contour1], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(largest_contour1)
            center_y = y + h // 2
            leftmost = (x, center_y)
            rightmost = (x + w, center_y)
            cv2.line(contours_image, leftmost, rightmost, (0, 0, 255), 2)
            cv2.line(contours_image, (x, center_y - 10), (x, center_y + 10), (0, 0, 255), 2)
            cv2.line(contours_image, (x + w, center_y - 10), (x + w, center_y + 10), (0, 0, 255), 2)
            diameter_cm = pixel_to_cm(w)
            size_class = get_size_class(diameter_cm)

        # Process the cleaned binary image
        binary_image_cleaned_no_text = cv2.cvtColor(binary_image_cleaned, cv2.COLOR_BGR2RGB)

        # Process Diameter X
        binary_image_cleaned = cv2.cvtColor(binary_image_cleaned, cv2.COLOR_GRAY2BGR)

        if contours1:
            largest_contour1 = max(contours1, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour1)

            # Diameter along X-axis
            diameter_x_pixels = w
            diameter_x_cm = pixel_to_cm(diameter_x_pixels)

            # Draw the line showing the X-axis diameter
            cv2.line(binary_image_cleaned, (x, y + h // 2), (x + w, y + h // 2), (0, 255, 255), 8)
            cv2.putText(binary_image_cleaned, f"Diameter X: {diameter_x_cm:.2f} cm", (x + 10, y + h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (0, 0, 255), 8, cv2.LINE_AA)

        # RGB Measurement Area
        image_with_box = original_image.copy()
        box_size = int(min(w, h) * 0.3)
        box_x = x + int(w / 2 - box_size / 2)
        box_y = y + int(h / 2 - box_size / 2)
        cv2.rectangle(image_with_box, (box_x, box_y), (box_x + box_size, box_y + box_size), (255, 0, 0), 2)

        points = [
            (box_x + int(box_size * 0.2), box_y + int(box_size * 0.2)),
            (box_x + int(box_size * 0.8), box_y + int(box_size * 0.2)),
            (box_x + int(box_size * 0.2), box_y + int(box_size * 0.8)),
            (box_x + int(box_size * 0.8), box_y + int(box_size * 0.8)),
            (box_x + int(box_size * 0.5), box_y + int(box_size * 0.5))
        ]

        point_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        colors = [tuple(original_image[p[1], p[0]]) for p in points]
        for p, c in zip(points, point_colors):
            cv2.circle(image_with_box, p, 5, c, -1)

        avg_color = np.mean(colors, axis=0).astype(int)
        print(f"Average RGB color: {tuple(avg_color)}")
        print(f"Diameter X: {diameter_x_cm:.2f} cm")

        # Create a folder for the current image
        image_folder = os.path.join(output_folder, filename.split('.')[0])
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)

        # Save processed images
        cv2.imwrite(os.path.join(image_folder, 'Diameter_X_Cleaned_Image.jpg'), binary_image_cleaned)
        cv2.imwrite(os.path.join(image_folder, 'RGB_Measurement_Area.jpg'), image_with_box)

        # Append data to the Excel list
        data.append([filename, diameter_x_cm, avg_color[0], avg_color[1], avg_color[2]])

# Save data to Excel file
csv_file = '/content/drive/MyDrive/Project/MangosteenData.csv'
df = pd.DataFrame(data, columns=["Image Name", "Diameter (cm)", "Red", "Green", "Blue"])
df.to_csv(csv_file, index=False)

print(f"csv file saved to {csv_file}")

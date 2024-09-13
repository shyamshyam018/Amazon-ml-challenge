import easyocr
import cv2
import matplotlib.pyplot as plt

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])  # Use 'en' for English

# Path to the image file (replace with your image path)
image_path = r'S:\Amazon-ML\downloaded_dataset\31EvJszFVfL.jpg'

# Read the image using OpenCV
image = cv2.imread(image_path)

# Perform OCR on the image using EasyOCR
results = reader.readtext(image_path)

# Draw bounding boxes around the detected text and display the image
for (bbox, text, prob) in results:
    # Extract the coordinates for the bounding box
    (top_left, top_right, bottom_right, bottom_left) = bbox
    top_left = tuple(map(int, top_left))
    bottom_right = tuple(map(int, bottom_right))

    # Draw a rectangle around the text
    cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)

    # Display the detected text in the command line
    print(f"Detected text: {text} (Confidence: {prob:.4f})")

    # Optionally: Display the text on the image
    cv2.putText(image, text, (top_left[0], top_left[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Display the image with bounding boxes
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')  # Turn off axis labels
plt.show()

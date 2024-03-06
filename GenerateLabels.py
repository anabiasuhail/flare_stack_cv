import torch
import cv2
from PIL import Image
import os
# from torchvision.models import yolov5


# Load the YOLOv5 model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the pre-trained YOLOv5 model
# model = yolov5('C:/Users/user/Downloads/videoFlare/ADNOC1200/yolov5s300Flame.pt')
model = torch.hub.load('C:/Users/user/Downloads/yolov5/yolov5/yolov5', 'custom', path='C:/Users/user/Downloads/videoFlare/ADNOC1200/yolov5s300Flame.pt', source='local')
# Set the model in evaluation mode
model.eval()



def run_yolo_detector(image_path, output_path):
    
    # Read the input image
    image = Image.open(image_path)

    # Perform object detection
    results = model(image)

    # Get the detections
    detections = results.pandas().xyxy[0]

    # Save the detections in YOLO label format
    with open(output_path, "w") as output_file:
        for _, detection in detections.iterrows():
            class_id = int(detection['class'])
            confidence = float(detection['confidence'])
            x_min, y_min, x_max, y_max = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
            print("x min : "+str(x_min)+" y min : "+str(y_min)+" x max : "+str(x_max)+" y max : "+str(y_max))
            width, height = (x_max - x_min)/image.width , (y_max - y_min)/image.height
            x_center, y_center = (x_min + x_max) / 2 / image.width, (y_min + y_max) / 2 / image.height
            print("x_center : "+str(x_center)+" y_center : "+str(y_center)+" width : "+str(width)+" height : "+str(height))

            # Write the detection to the output file
            output_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

    print("Detections saved successfully.")

# # Usage example
# image_path = "C:/Users/user/Downloads/videoFlare/ADNOC1200/frame800.jpg"
# output_path = "C:/Users/user/Downloads/videoFlare/ADNOC1200/Generated_Detections.txt"

# run_yolo_detector(image_path, output_path)


def iterate_images_in_folder(folder_path, output_folder):
    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more file extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Get the path to the current image and the corresponding output file path
            image_path = os.path.join(folder_path, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            # Call the run_yolo_detector function
            run_yolo_detector(image_path, output_path)
            print(f"Processed {filename}")

# Usage example
folder_path = 'C:/Users/user/Downloads/videoFlare/ADNOC1200/ImagesForAutoAnnotation'
output_folder = 'C:/Users/user/Downloads/videoFlare/ADNOC1200/autoannotationOutput'

iterate_images_in_folder(folder_path, output_folder)

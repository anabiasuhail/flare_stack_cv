import tkinter as tk
import torch
import math
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, scrolledtext
import os
import sys
from tkinter import ttk
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from ultralytics import SAM
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def run_functions():
    iterate_images_in_folder()
    draw_yolo_labels()
    open_results_visualizer_window()


def open_results_visualizer_window():
    # Paths of folders
    folder_path = entry_dir1.get()
    output_folder = entry_dir2.get()

    detection_images_path = os.path.join(output_folder, "Detection_Images")
    segmentation_visualized_path1 = os.path.join(output_folder, "Segmentation_Visualized_flame")
    segmentation_visualized_path2 = os.path.join(output_folder, "Segmentation_Visualized_smoke")
    value_path1=os.path.join(output_folder, "flame_size")
    value_path2=os.path.join(output_folder, "smoke_size")
    value_path3=os.path.join(output_folder, "flame_smoke_ratio")
    value_path4=os.path.join(output_folder, "flame_orientation")
    # Function to get the first image from a folder
    def get_first_image(folder):
        for file in os.listdir(folder):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp','.txt')):
                return file
        return None  # No image found

    def find_matching_image(folder, partial_filename):
        if not partial_filename:
            return None
        for file in os.listdir(folder):
            if partial_filename.lower() in file.lower() and file.lower().endswith(
                    ('.png', '.jpg', '.jpeg', '.gif', '.bmp','.txt')):
                return os.path.join(folder, file)
        return None

    # Function to get a list of all image files in a folder
    def get_all_images(folder):
        return [file for file in os.listdir(folder) if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    # Function to update the displayed images
    def update_images(index):

        # Update original image
        original_image_path = os.path.join(folder_path, all_images[index])
        original_image = load_image(original_image_path, max_size=(400, 300))
        canvas1.itemconfig(image_on_canvas1, image=original_image)

        # Update detection image
        detection_image_path = os.path.join(detection_images_path, all_images[index])
        detection_image = load_image(detection_image_path, max_size=(400, 300))
        canvas2.itemconfig(image_on_canvas2, image=detection_image)


        segmentation_flame_path = find_matching_image(segmentation_visualized_path1,
                                                      os.path.splitext(all_images[index])[0])
        segmentation_flame = load_image(segmentation_flame_path, max_size=(400, 300))
        canvas3.itemconfig(image_on_canvas3, image=segmentation_flame)
        # Update segmentation image
        segmentation_smoke_path = find_matching_image(segmentation_visualized_path2,
                                                      os.path.splitext(all_images[index])[0])
        segmentation_smoke = load_image(segmentation_smoke_path, max_size=(400, 300))
        flame_size_pth = find_matching_image(value_path1, os.path.splitext(all_images[index])[
            0])
        smoke_size_pth = find_matching_image(value_path2, os.path.splitext(all_images[index])[
            0])
        flame_smoke_ratio_pth = find_matching_image(value_path3, os.path.splitext(all_images[index])[
            0])
        flame_orientation_pth = find_matching_image(value_path4, os.path.splitext(all_images[index])[
            0])
        value1 = []
        value2 = []
        value3 = []
        value4 = []

        with open(flame_size_pth, 'r') as file:
            lines = file.readlines()
        for line in lines:
            value = line.strip()  # 去除行尾的换行符和空格
            value1.append(value)
        with open(smoke_size_pth, 'r') as file:
            lines = file.readlines()
        for line in lines:
            value = line.strip()  # 去除行尾的换行符和空格
            value2.append(value)
        with open(flame_smoke_ratio_pth, 'r') as file:
            lines = file.readlines()
        for line in lines:
            line = line.strip()
            values = line.split(' ')
            for value in values:
                value3.append(value)
        with open(flame_orientation_pth, 'r') as file:
            lines = file.readlines()
        for line in lines:
            value = line.strip()  # 去除行尾的换行符和空格
            value4.append(value)
        canvas4.itemconfig(image_on_canvas4, image=segmentation_smoke)
        ax.clear()
        x_value = ['flame_image_ratio', 'smoke_image_ratio', 'smoke_area_ratio']
        y_value = [float(value3[0]), float(value3[1]), float(value3[2])]
        ax.bar(x_value, y_value)
        ax.tick_params(axis='both', labelsize=8)
        ax.set_title('flame_smoke_ratio')
        ax.set_xlabel('name')
        ax.set_ylabel('ratio_value')
        canvas5.draw()

        label5.config(text="flame size:" + value1[0]+"px"+"  "+"smoke size:" + value2[0])
        #label5 = tk.Label(results_window, text="flame size:" + str(value1[0])+"px"+"        "+"smoke size:" + str(value2[0]))
        #label6 = tk.Label(results_window, text="smoke size:" + str(value2[0]))
        #label6.grid(row=1, column=2)
        #label6.config(text="flame_smoke_ratio:" + value3[0]+"\n"+"flame size:" + value1[0]+"px"+"  "+"smoke size:" + value2[0]+"\n"+"flame's angle:" + value4[0])
        label6.config(text="smoke_area_ratio: " + value3[2])
        #label7 = tk.Label(results_window, text="flame_smoke_ratio:" + str(value3[0]))
        #label7.grid(row=1, column=2)
        label7.config(text="flame's angle: " + value4[0]+" (Rotate clockwise on the x-negative axis)")
        #label8 = tk.Label(results_window, text="flame's angle:" + str(value4[0]))
        #label8.grid(row=3, column=2)

        # Update references to prevent garbage collection
        results_window.images = [original_image, detection_image, segmentation_flame,segmentation_smoke]

    #def update_label():
        # 更新标签的文本
        #label.config(text="New Value")

    # 创建一个 Tkinter 窗口

    # 创建一个标签

    # 创建一个按钮，点击按钮时更新标签的文本
    # Navigation functions
    def next_image():
        nonlocal current_index
        current_index = (current_index + 1) % len(all_images)
        update_images(current_index)

    def previous_image():
        nonlocal current_index
        current_index = (current_index - 1) % len(all_images)
        update_images(current_index)

    # Get the first image from the source folder
    first_image_filename = get_first_image(folder_path)
    original_image_path = os.path.join(folder_path, first_image_filename) if first_image_filename else None

    # Construct paths for the corresponding images in the output folders
    detection_image_path = os.path.join(detection_images_path, first_image_filename) if first_image_filename else None

    # Find a matching image in the segmentation_visualized_path
    segmentation_flame_path = find_matching_image(segmentation_visualized_path1, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    segmentation_smoke_path = find_matching_image(segmentation_visualized_path2, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    flame_size_pth = find_matching_image(value_path1, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    smoke_size_pth = find_matching_image(value_path2, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    flame_smoke_ratio_pth=find_matching_image(value_path3, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None
    flame_orientation_pth= find_matching_image(value_path4, os.path.splitext(first_image_filename)[
        0]) if first_image_filename else None


    # Create a new Toplevel window
    results_window = tk.Toplevel(root)
    results_window.title("Results Visualizer")
    results_window.geometry("1250x800")
   # results_window.columnconfigure(0, weight=1)
    #results_window.columnconfigure(1, weight=1)
    #results_window.rowconfigure(0, weight=1)
    #results_window.rowconfigure(1, weight=1)
    x_value=[]
    y_value=[]
    value1=[]
    value2=[]
    value3=[]
    value4=[]

    # Load and display images on canvases
    with open(flame_size_pth, 'r') as file:
        lines = file.readlines()
    for line in lines:
        value = line.strip()  # 去除行尾的换行符和空格
        value1.append(value)
    with open(smoke_size_pth, 'r') as file:
        lines = file.readlines()
    for line in lines:
        value = line.strip()  # 去除行尾的换行符和空格
        value2.append(value)
    with open(flame_smoke_ratio_pth, 'r') as file:
        lines = file.readlines()
    for line in lines:
        line= line.strip()
        values = line.split(' ')
        for value in values:
            value3.append(value)
    with open(flame_orientation_pth, 'r') as file:
        lines = file.readlines()
    for line in lines:
        value = line.strip()  # 去除行尾的换行符和空格
        value4.append(value)
    if original_image_path:
        original_image = load_image(original_image_path, max_size=(400, 300))
        canvas1 = tk.Canvas(results_window, width=400, height=300)
        canvas1.create_image(20, 20, anchor=tk.NW, image=original_image)
        #canvas1.create_text(200, 290, anchor='n', text="Original Image",font=('Arial', 12), fill='white', justify='center')
        canvas1.grid(row=0, column=0, padx=0, pady=0)
        label1 = tk.Label(results_window, text="Original Image")
        label1.grid(row=1, column=0,padx=0,pady=0)

    if os.path.exists(detection_image_path):
        detection_image = load_image(detection_image_path, max_size=(400, 300))
        canvas2 = tk.Canvas(results_window, width=400, height=300)
        canvas2.create_image(20, 20, anchor=tk.NW, image=detection_image)
        #canvas2.create_text(200, 290, anchor='n', text="Detection",font=('Arial', 12), fill='white', justify='center')
        canvas2.grid(row=0, column=1, padx=0, pady=0)
        label2 = tk.Label(results_window, text="Detection")
        #canvas2.create_text(0, 300, anchor='sw', text="Description 1")
        label2.grid(row=1, column=1,padx=0,pady=0)

    if os.path.exists(segmentation_flame_path):
        segmentation_flame = load_image(segmentation_flame_path, max_size=(400, 300))
        canvas3 = tk.Canvas(results_window, width=400, height=300)
        canvas3.create_image(20, 20, anchor=tk.NW, image=segmentation_flame)
        #canvas3.create_text(200, 290, anchor='n', text="Segmentation_flame",font=('Arial', 12), fill='white', justify='center')
        canvas3.grid(row=2, column=0, padx=0, pady=0)
        #canvas3.create_text(0, 300, anchor='sw', text="Description 1")
        label3 = tk.Label(results_window, text="Segmentation_flame")
        label3.grid(row=3, column=0,padx=0,pady=0)
    if os.path.exists(segmentation_smoke_path):
        segmentation_smoke = load_image(segmentation_smoke_path, max_size=(400, 300))
        canvas4 = tk.Canvas(results_window, width=400, height=300)
        canvas4.create_image(20, 20, anchor=tk.NW, image=segmentation_smoke)
        canvas4.grid(row=2, column=1, padx=0, pady=0)

        label4 = tk.Label(results_window, text="Segmentation_smoke")
        #canvas4.create_text(0, 300, anchor='sw', text="Description 1"
        label4.grid(row=3, column=1,padx=0,pady=0)
    canvas5 = FigureCanvasTkAgg(Figure(figsize=(3.8, 3), dpi=100), master=results_window)
    canvas5.draw()
    canvas5.get_tk_widget().grid(row=0, column=2,padx=0,pady=0)
    ax =canvas5.figure.add_subplot(111)
    x_value=['flame_image_ratio','smoke_image_ratio','smoke_area_ratio']
    y_value=[float(value3[0]),float(value3[1]),float(value3[2])]
    ax.bar(x_value, y_value)
    ax.tick_params(axis='both', labelsize=8)  # 设置刻度标签的字体大小为12
    ax.set_title('flame_smoke_ratio')
    ax.set_xlabel('name')
    ax.set_ylabel('ratio_value')
    if os.path.exists(flame_size_pth)&os.path.exists(smoke_size_pth):
        label5=tk.Label(results_window,text="flame size:"+value1[0]+"px"+"  "+"smoke size:"+value2[0]+"px" )
        label5.grid(row=2, column=2,padx=0,pady=0)

    #if os.path.exists(smoke_size_pth):
        #label6=tk.Label(results_window,text="smoke size:"+str(value2[0]) )
        #label6.grid(row=1, column=2)
    if os.path.exists(flame_smoke_ratio_pth):
        label6=tk.Label(results_window,text="smoke_area_ratio:"+value3[2] )
        label6.grid(row=1, column=2,padx=0,pady=0)
    if os.path.exists(flame_orientation_pth):
        label7=tk.Label(results_window,text="flame's angle: "+value4[0]+" (Rotate clockwise on the x-negative axis)" )
        label7.grid(row=3, column=2,padx=0,pady=0)
    # 将画布布局到GUI上
    # Create and position the description labels for each box
   #

    #label2 = tk.Label(results_window, text="Detection")
    #label2.grid(row=1, column=1,padx=5, pady=5)

   # label3 = tk.Label(results_window, text="Segmentation_flame")
    #label3.grid(row=2, column=0,padx=10, pady=10)
    #label4 = tk.Label(results_window, text="Segmentation_smoke")
    #label4.grid(row=2, column=1,padx=10, pady=10)
    image_on_canvas1 = canvas1.create_image(20, 20, anchor=tk.NW, image=original_image)
    image_on_canvas2 = canvas2.create_image(20, 20, anchor=tk.NW, image=detection_image)
    image_on_canvas3 = canvas3.create_image(20, 20, anchor=tk.NW, image=segmentation_flame)
    image_on_canvas4 = canvas4.create_image(20, 20, anchor=tk.NW, image=segmentation_smoke)

    # Get all images in the folder
    all_images = get_all_images(folder_path)
    current_index = 0  # Index of the currently displayed image
    # Create navigation buttons
    next_button = ttk.Button(results_window, text="Next Image", command=next_image)
    next_button.grid(row=4, column=1,padx=15,pady=15)
    #next_button.pack()

    prev_button = ttk.Button(results_window, text="Previous Image", command=previous_image)
    prev_button.grid(row=4, column=0,padx=15,pady=15)
    #prev_button.pack()

    ## ADDED TO HERE ##

    # Keep references to the images to prevent garbage collection
    results_window.images = [original_image, detection_image, segmentation_flame,segmentation_smoke]


def draw_yolo_labels():
    # Create a subfolder for detection images
    image_folder = entry_dir1.get()
    output_folder = entry_dir2.get()

    label_folder = os.path.join(output_folder, "Raw_Detection_labels")
    os.makedirs(label_folder, exist_ok=True)  # Ensure label folder exists

    detection_images_folder = os.path.join(output_folder, "Detection_Images")
    os.makedirs(detection_images_folder, exist_ok=True)

    # Iterate over all image files in the image folder
    for image_file in os.listdir(image_folder):
        # Check for image file extension
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_file)
            label_file = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(label_folder, label_file)

            # Check if the corresponding label file exists
            if not os.path.exists(label_path):
                print(f"Label file for {image_file} does not exist. Skipping.")
                continue

            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read image {image_file}. Skipping.")
                continue

            # Get image dimensions
            img_height, img_width = image.shape[:2]

            # Read YOLO labels from the label file
            with open(label_path, 'r') as file:
                lines = file.readlines()

                for line in lines:
                    # Parse YOLO format (class_id, x_center, y_center, width, height)
                    class_id, x_center, y_center, width, height = map(float, line.split())

                    # Convert normalized positions to pixel values
                    x_center, y_center, width, height = (x_center * img_width, y_center * img_height,
                                                         width * img_width, height * img_height)
                    x_min = int(x_center - (width / 2))
                    y_min = int(y_center - (height / 2))

                    # Draw rectangle
                    if int(class_id) == 0:
                        cv2.rectangle(image, (x_min, y_min), (x_min + int(width), y_min + int(height)), (0, 0, 255), 2)
                        cv2.putText(image, str("Flame"), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (0, 0, 255), 2)
                    else:
                        cv2.rectangle(image, (x_min, y_min), (x_min + int(width), y_min + int(height)), (255, 0, 0), 2)
                        cv2.putText(image, str("Smoke"), (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, (255, 0, 0), 2)

            # Adjust the output path to save in the Detection_Images subfolder
            output_image_path = os.path.join(detection_images_folder, image_file)
            cv2.imwrite(output_image_path, image)
    print(f"Processed and saved Detection Images and raw labels.")
    print(f"Processed and saved Segmentation Images and raw labels.")


# Usage example:
# draw_yolo_labels('path_to_image_folder', 'path_to_label_folder', 'path_to_output_folder')


def open_new_window():
    # Create a new Toplevel window
    new_window = tk.Toplevel(root)
    new_window.title("About This Tool")
    new_window.geometry("600x160")  # Adjusted the size of the window

    # Multiline text for the label
    text = """This tool was developed under the project titled "Vision-based Flare Analytics"

     in a collaboration between Khalifa University and ADNOC.

The tool was developed by Muaz Al Radi (KU) and Mu Xing (KU),

 under the supervision of Prof. Naoufel Werghi."""

    # Add a label with multiline text to the new window
    label = tk.Label(new_window, text=text, justify=tk.CENTER)
    label.pack(padx=10, pady=20)


# to adjust, to add the two logos on top of the text in the "about" widnow
# def open_new_window():
#     # Create a new Toplevel window
#     new_window = tk.Toplevel(root)
#     new_window.title("About This Tool")
#     new_window.geometry("600x250")  # Adjusted the size of the window

#     # Load images
#     kustar_logo = load_image("KUSTAR_Logo.jpg")
#     adnoc_logo = load_image("adnoc_logo.jpg")

#     # Place images in labels
#     kustar_label = tk.Label(new_window, image=kustar_logo)
#     kustar_label.grid(row=0, column=0, padx=10, pady=10)

#     adnoc_label = tk.Label(new_window, image=adnoc_logo)
#     adnoc_label.grid(row=0, column=1, padx=10, pady=10)

#     # Multiline text for the label
#     text = """This tool was developed under the project titled "Vision-based Flare Analytics"
#      in a collaboration between Khalifa University and ADNOC.
#     The tool was developed by Muaz Al Radi (PhD student, KU),
#     under the supervision of Prof. Naoufel Werghi."""

#     # Add a label with multiline text to the new window
#     label = tk.Label(new_window, text=text, justify=tk.CENTER)
#     label.grid(row=1, column=0, columnspan=2, padx=10, pady=20)


def select_directory1():
    directory1 = filedialog.askdirectory()
    entry_dir1.delete(0, tk.END)
    entry_dir1.insert(0, directory1)


def select_directory2():
    directory2 = filedialog.askdirectory()
    entry_dir2.delete(0, tk.END)
    entry_dir2.insert(0, directory2)


def save_directories():
    dir1 = entry_dir1.get()
    dir2 = entry_dir2.get()
    # Here you can save these directories to a file or use them further in the code
    print("Directory 1:", dir1)
    print("Directory 2:", dir2)


class PrintLogger:
    def __init__(self, textbox):
        self.textbox = textbox

    def write(self, text):
        self.textbox.configure(state='normal')  # Enable the log box to insert text
        self.textbox.insert(tk.END, text)  # Write text to the textbox
        self.textbox.see(tk.END)  # Scroll to end
        self.textbox.configure(state='disabled')  # Disable the log box to prevent user editing

    def flush(self):
        pass


def run_yolo_detector(image_path, det_model_path, sam_model_path, output_path):
    # Load the models
    det_model = YOLO(det_model_path)
    sam_model = SAM(sam_model_path)

    # Create subfolders for output
    raw_detection_labels_path = os.path.join(output_path, "Raw_Detection_labels")
    raw_segmentation_labels_path1 = os.path.join(output_path, "Raw_Segmentation_flame_labels")
    raw_segmentation_labels_path2= os.path.join(output_path, "Raw_Segmentation_smoke_labels")
    segmented_flame_path = os.path.join(output_path, "Segmentation_Visualized_flame")
    segmented_smoke_path = os.path.join(output_path, "Segmentation_Visualized_smoke")
    flame_size_path = os.path.join(output_path, "flame_size")
    smoke_size_path = os.path.join(output_path, "smoke_size")
    flame_smoke_ratio_path=os.path.join(output_path, "flame_smoke_ratio")
    flame_orientation_path=os.path.join(output_path, "flame_orientation")

    #empty_seg_filename1 = os.path.splitext(os.path.basename(image_path))[0] + "_empty.txt"
    os.makedirs(raw_detection_labels_path, exist_ok=True)
    os.makedirs(raw_segmentation_labels_path1, exist_ok=True)
    os.makedirs(raw_segmentation_labels_path2, exist_ok=True)
    os.makedirs(segmented_flame_path, exist_ok=True)
    os.makedirs(segmented_smoke_path, exist_ok=True)
    os.makedirs(flame_size_path, exist_ok=True)
    os.makedirs(smoke_size_path, exist_ok=True)
    os.makedirs(flame_smoke_ratio_path, exist_ok=True)
    os.makedirs(flame_orientation_path, exist_ok=True)

    # Read the input image
    image = Image.open(image_path)
    img_width=image.width
    img_height=image.height

    # Convert PIL image to OpenCV format
    image_cv = np.array(image)
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

    # Perform object detection
    results = det_model(image, stream=True)
    empty_seg_filename1="none.txt"

    # Open the output file for YOLO formatted data
    yolo_labels_filename = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    value_filename=os.path.splitext(os.path.basename(image_path))[0]
    with open(os.path.join(raw_detection_labels_path, yolo_labels_filename), "w") as output_file:
        masks_found = False
        for result in results:
            flame_size=0
            smoke_size=0
            smoke_area_ratio=0
            flame_image_ratio=0
            smoke_image_ratio=0
            flame_orientation=0
            box_max=0
            temp1=0
            temp2=0
            boxes = result.boxes.xyxy  # Bounding boxes
            class_ids = result.boxes.cls.int().tolist()  # Class IDs

            # Check if there are any detections
            if boxes.numel() == 0:
                # No masks found, save an empty black image
                img_height, img_width = image.size
                empty_img = np.zeros((img_width, img_height, 3), dtype=np.uint8)
                empty_img_filename = os.path.splitext(os.path.basename(image_path))[0] + "_empty.png"
                empty_img_path1 = os.path.join(segmented_flame_path, empty_img_filename)
                empty_img_path2 = os.path.join(segmented_smoke_path, empty_img_filename)
                cv2.imwrite(empty_img_path1, empty_img)
                cv2.imwrite(empty_img_path2, empty_img)

                # Also save an empty .txt file for segmentation labels
                empty_seg_filename = os.path.splitext(os.path.basename(image_path))[0] + "_segmentation_empty.txt"
                empty_seg_path1 = os.path.join(raw_segmentation_labels_path1, empty_seg_filename)
                empty_seg_path2 = os.path.join(raw_segmentation_labels_path2, empty_seg_filename)
                empty_seg_filename1 = os.path.splitext(os.path.basename(image_path))[0] + "_empty.txt"
                empty_seg_path3 = os.path.join(flame_size_path, empty_seg_filename1)
                empty_seg_path4 = os.path.join(smoke_size_path, empty_seg_filename1)
                empty_seg_path5 = os.path.join(flame_smoke_ratio_path, empty_seg_filename1)
                empty_seg_path6 = os.path.join(flame_orientation_path, empty_seg_filename1)
                open(empty_seg_path1, 'w').close()
                open(empty_seg_path2, 'w').close()
                with open(empty_seg_path3, "w") as f1:
                    f1.write(f" {flame_size:.2f} \n")
                with open(empty_seg_path4,"w") as f2:
                    f2.write(f" {smoke_size:.2f} \n")
                with open(empty_seg_path5, "w") as f3:
                    f3.write(f" {flame_image_ratio:.2f} {smoke_image_ratio:.2f} {smoke_area_ratio:.2f} \n")
                with open(empty_seg_path6, "w") as f4:
                    f4.write(f" {flame_orientation:.2f} \n")

                continue  # No detections, continue with the next result

            # Iterate through each detected object
            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box.tolist())
                class_id = class_ids[i]
                img_width, img_height = image.size
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                # Write the detection in YOLO format to the output file
                output_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

                if class_id==0:
                    flame_size = (x_max - x_min)*(y_max - y_min)+flame_size
                    box_size=(x_max - x_min)*(y_max - y_min)
                    # calculate slope
                    if box_size>box_max:
                        if x_min - x_max != 0:
                            slope = (y_max - y_min) / (x_max - x_min)
                        else:
                            slope = float('inf')  # other situation

                            # calculate angle
                        angle_rad = math.atan(slope)

                        # convert
                        flame_orientation = math.degrees(angle_rad)

                    temp1=temp1+1
                    # Apply SAM to the image
                    segmented_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                                                 labels=[class_id])

                    # Check if masks are available in the result
                    if segmented_result[0].masks is not None:
                        masks = segmented_result[0].masks.data.cpu().numpy()
                        masks_found = True  # Indicate that at least one mask was found

                        # Process each mask
                        for i, mask in enumerate(masks):
                            # Ensure mask is 2D and convert to uint8
                            mask_2d = mask.squeeze()  # Remove any extra dimensions
                            mask_cv = np.uint8(mask_2d * 255)

                            # Find contours of the mask
                            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Construct filename for raw segmentation labels
                            raw_segmentation_filename = os.path.splitext(os.path.basename(image_path))[
                                                            0] + f"_segmentation_{i}.txt"
                            raw_segmentation_path = os.path.join(raw_segmentation_labels_path1,
                                                                 raw_segmentation_filename)

                            # Open the segmentation label file
                            with open(raw_segmentation_path, "w") as seg_file:
                                seg_file.write(f"{class_id}")
                                for cnt in contours:
                                    for point in cnt:
                                        x, y = point[0]
                                        # Normalize the coordinates
                                        normalized_x = x / img_width
                                        normalized_y = y / img_height
                                        seg_file.write(f" {normalized_x:.6f} {normalized_y:.6f}")
                                        # seg_file.write(f" {x} {y}")
                                seg_file.write("\n")

                            # Apply the mask to the image
                            segmented_img = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)

                            # Construct the filename for the segmented image
                            segmented_filename = os.path.splitext(os.path.basename(image_path))[
                                                     0] + f"_segmented_{i}.png"
                            segmented_flame_path = os.path.join(segmented_flame_path, segmented_filename)

                            # Save the segmented image
                            cv2.imwrite(segmented_flame_path, segmented_img)
                if class_id==1:
                    smoke_size=(x_max-x_min)*(y_max - y_min)+smoke_size
                    temp2=temp2+1
                    # Apply SAM to the image
                    segmented_result = sam_model(image, points=[(x_min + x_max) / 2, (y_min + y_max) / 2],
                                                 labels=[class_id])

                    # Check if masks are available in the result
                    if segmented_result[0].masks is not None:
                        masks = segmented_result[0].masks.data.cpu().numpy()
                        masks_found = True  # Indicate that at least one mask was found

                        # Process each mask
                        for i, mask in enumerate(masks):
                            # Ensure mask is 2D and convert to uint8
                            mask_2d = mask.squeeze()  # Remove any extra dimensions
                            mask_cv = np.uint8(mask_2d * 255)

                            # Find contours of the mask
                            contours, _ = cv2.findContours(mask_cv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                            # Construct filename for raw segmentation labels
                            raw_segmentation_filename = os.path.splitext(os.path.basename(image_path))[
                                                            0] + f"_segmentation_{i}.txt"
                            raw_segmentation_path = os.path.join(raw_segmentation_labels_path2,
                                                                 raw_segmentation_filename)

                            # Open the segmentation label file
                            with open(raw_segmentation_path, "w") as seg_file:
                                seg_file.write(f"{class_id}")
                                for cnt in contours:
                                    for point in cnt:
                                        x, y = point[0]
                                        # Normalize the coordinates
                                        normalized_x = x / img_width
                                        normalized_y = y / img_height
                                        seg_file.write(f" {normalized_x:.6f} {normalized_y:.6f}")
                                        # seg_file.write(f" {x} {y}")
                                seg_file.write("\n")

                            # Apply the mask to the image
                            segmented_img = cv2.bitwise_and(image_cv, image_cv, mask=mask_cv)

                            # Construct the filename for the segmented image
                            segmented_filename = os.path.splitext(os.path.basename(image_path))[
                                                     0] + f"_segmented_{i}.png"
                            segmented_smoke_path = os.path.join(segmented_smoke_path, segmented_filename)

                            # Save the segmented image
                            cv2.imwrite(segmented_smoke_path, segmented_img)

        flame_image_ratio=flame_size/(img_width*img_height)
        smoke_image_ratio=smoke_size/(img_width*img_height)
        smoke_area_ratio=smoke_size/(flame_size+smoke_size)
        #flame_smoke_ratio = flame_size / (smoke_size + 1e-6)

        if temp2==0:
            img_height, img_width = image.size
            empty_img = np.zeros((img_width, img_height, 3), dtype=np.uint8)
            empty_img_filename = os.path.splitext(os.path.basename(image_path))[0] + "_empty.png"
            empty_img_path2 = os.path.join(segmented_smoke_path, empty_img_filename)
            cv2.imwrite(empty_img_path2, empty_img)
            # Also save an empty .txt file for segmentation labels
            empty_seg_filename = os.path.splitext(os.path.basename(image_path))[0] + "_segmentation_empty.txt"
            empty_seg_path2 = os.path.join(raw_segmentation_labels_path2, empty_seg_filename)
            open(empty_seg_path2, 'w').close()
            #flame_smoke_ratio=1
            flame_image_ratio=1
            smoke_image_ratio=1
            smoke_area_ratio=1

        if not (temp1 or temp2):
            flame_image_ratio=0
            smoke_image_ratio=0
            smoke_area_ratio=0

            #flame_smoke_ratio = 0



        if temp1 or temp2:
        #if yolo_labels_filename.lower() not in empty_seg_filename1.lower():
            with open(os.path.join(flame_size_path, yolo_labels_filename), "w") as output_file1:
                output_file1.write(f" {flame_size:.2f} \n")
            with open(os.path.join(smoke_size_path, yolo_labels_filename), "w") as output_file2:
                output_file2.write(f" {smoke_size:.2f} \n")
            with open(os.path.join(flame_smoke_ratio_path, yolo_labels_filename), "w") as output_file3:
                output_file3.write(f" {flame_image_ratio:.2f} {smoke_image_ratio:.2f} {smoke_area_ratio:.2f} \n")
            with open(os.path.join(flame_orientation_path, yolo_labels_filename), "w") as output_file4:
                output_file4.write(f" {flame_orientation:.2f} \n")



def iterate_images_in_folder():
    det_model_path = 'best1.pt'  # det_model_path = 'E:\\GUI\\GUI\\bestYolov8x.pt'
    sam_model_path = 'sam_b.pt'

    # Iterate over all files in the folder
    folder_path = entry_dir1.get()
    output_folder = entry_dir2.get()

    # Count processed images
    counter = 0

    for filename in os.listdir(folder_path):
        # Check if the file is an image (you can add more file extensions if needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            counter += 1
            image_path = os.path.join(folder_path, filename)

            # Call the run_yolo_detector function
            run_yolo_detector(image_path, det_model_path, sam_model_path, output_folder)
            print(f"Processed {filename}")

    print("----------------------------------------")
    print(f"Processed a total of {counter} images.")


# Function to load an image and return a PhotoImage object
def load_image(path, max_size=(150, 150)):
    image = Image.open(path)
    image.thumbnail(max_size, Image.ANTIALIAS)
    return ImageTk.PhotoImage(image)


# Create the main window
root = tk.Tk()
root.title("Autolabelling Tool GUI")

# Simply set the theme
root.tk.call("source", "azure.tcl")
root.tk.call("set_theme", "dark")

# Load images
kustar_logo = load_image("KUSTAR_Logo.jpg")
adnoc_logo = load_image("adnoc_logo.jpg")

# Place images in labels
kustar_label = tk.Label(root, image=kustar_logo)
kustar_label.grid(row=0, column=0, padx=10, pady=10)

adnoc_label = tk.Label(root, image=adnoc_logo)
adnoc_label.grid(row=0, column=1, padx=10, pady=10)

# Create entry widgets to show selected directories
entry_dir1 = ttk.Entry(root, width=50)
entry_dir1.grid(row=1, column=1, padx=10, pady=10)

entry_dir2 = ttk.Entry(root, width=50)
entry_dir2.grid(row=2, column=1, padx=10, pady=10)

# Create buttons to select directories
button_dir1 = ttk.Button(root, text="Select Image Directory", command=select_directory1)
button_dir1.grid(row=1, column=0, padx=10, pady=10)
button_dir2 = ttk.Button(root, text="Select Output Directory", command=select_directory2)
button_dir2.grid(row=2, column=0, padx=10, pady=10)

# Create a button to save the selected directories
button_save = ttk.Button(root, text="Start Autolabelling", command=run_functions)
button_save.grid(row=3, column=1, padx=10, pady=10)  # button_save.grid(row=3, column=0, columnspan=2, pady=10)

# Create a button to open a new window
button_new_window = ttk.Button(root, text="About This Tool", command=open_new_window)
button_new_window.grid(row=3, column=0, padx=10, pady=10)  # Place it next to the "Start Autolabelling" button

# Create a ScrolledText widget for logs
log_box = scrolledtext.ScrolledText(root, state='disabled', height=10)
log_box.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky='ew')

# Redirect stdout to log box
pl = PrintLogger(log_box)
sys.stdout = pl
# Start the main loop
root.mainloop()
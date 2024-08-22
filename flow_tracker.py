import cv2
import math
import matplotlib.pyplot as plt
import csv
import tkinter as tk
from tkinter import filedialog, simpledialog, ttk
import os

# Function to initialize the object tracker
def create_object_tracker(tracker_type):
    if tracker_type == 'kcf':
        return cv2.TrackerKCF_create()
    elif tracker_type == 'csrt':
        return cv2.TrackerCSRT_create()
    elif tracker_type == 'mosse':
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_type == 'MIL':
        return cv2.TrackerMIL_create()
    elif tracker_type == 'boost':
        return cv2.legacy.TrackerBoosting_create()
    elif tracker_type == 'TLD':
        return cv2.legacy.TrackerTLD_create()
    elif tracker_type == 'mf':
        return cv2.legacy.TrackerMedianFlow_create()
    
def apply_gaussian_blur(frame, kernel_size=(51, 51)):
    # Apply Gaussian Blur to the frame
    return cv2.GaussianBlur(frame, kernel_size, 0)

def double_difference(previous_frame, current_frame, prev2_frame):
    # Calculate the absolute difference between the current frame and the previous frame
    diff_frame1 = cv2.absdiff(current_frame, previous_frame)

    # Calculate the absolute difference between the current frame and the frame just before the previous frame
    diff_frame2 = cv2.absdiff(current_frame, prev2_frame)

    # Combine the two difference frames using bitwise OR operation
    return cv2.bitwise_or(diff_frame1, diff_frame2)

def get_user_parameters():
    parameters = {}

    def submit():
        parameters['desired_width'] = int(desired_width_entry.get())
        parameters['fps'] = int(fps_entry.get())
        parameters['frame_skip'] = int(frame_skip_entry.get())
        parameters['channel_width'] = float(channel_width_entry.get())
        pixel_width = float(pixel_width_entry.get())
        parameters['pixel_distance'] = parameters['channel_width'] / pixel_width
        parameters['tracker_type'] = tracker_type_combo.get()
        parameters['start_time'] = float(start_time_entry.get())
        parameters['end_time'] = float(end_time_entry.get())
        parameters['output_directory'] = filedialog.askdirectory(title="Select Output Directory")
        parameters['filename'] = filename_entry.get()

        if not parameters['output_directory'] or not parameters['filename']:
            tk.messagebox.showerror("Error", "Output directory or filename not provided. Exiting...")
            root.destroy()
            return

        root.quit()

    root = tk.Tk()
    root.title("Enter Parameters")

    ttk.Label(root, text="Desired Video Width (pixels):").grid(row=0, column=0)
    desired_width_entry = ttk.Entry(root)
    desired_width_entry.insert(0, "1920")
    desired_width_entry.grid(row=0, column=1)

    ttk.Label(root, text="Frame Rate (fps):").grid(row=1, column=0)
    fps_entry = ttk.Entry(root)
    fps_entry.insert(0, "60")
    fps_entry.grid(row=1, column=1)

    ttk.Label(root, text="Number of Frames to Skip:").grid(row=2, column=0)
    frame_skip_entry = ttk.Entry(root)
    frame_skip_entry.insert(0, "2")
    frame_skip_entry.grid(row=2, column=1)

    ttk.Label(root, text="Channel Width (mm):").grid(row=3, column=0)
    channel_width_entry = ttk.Entry(root)
    channel_width_entry.insert(0, "0.25")
    channel_width_entry.grid(row=3, column=1)

    ttk.Label(root, text="Pixel Width (pixels):").grid(row=4, column=0)
    pixel_width_entry = ttk.Entry(root)
    pixel_width_entry.insert(0, "11")
    pixel_width_entry.grid(row=4, column=1)

    ttk.Label(root, text="Tracker Type:").grid(row=5, column=0)
    tracker_type_combo = ttk.Combobox(root, values=['kcf', 'csrt', 'mosse', 'MIL', 'boost', 'TLD', 'mf'])
    tracker_type_combo.set('boost')
    tracker_type_combo.grid(row=5, column=1)

    ttk.Label(root, text="Start Time (seconds):").grid(row=6, column=0)
    start_time_entry = ttk.Entry(root)
    start_time_entry.insert(0, "0")
    start_time_entry.grid(row=6, column=1)

    ttk.Label(root, text="End Time (seconds):").grid(row=7, column=0)
    end_time_entry = ttk.Entry(root)
    end_time_entry.insert(0, "10")
    end_time_entry.grid(row=7, column=1)

    ttk.Label(root, text="Output Filename (without extension):").grid(row=8, column=0)
    filename_entry = ttk.Entry(root)
    filename_entry.insert(0, "output")
    filename_entry.grid(row=8, column=1)

    submit_button = ttk.Button(root, text="Submit", command=submit)
    submit_button.grid(row=9, columnspan=2)

    root.mainloop()
    root.destroy()

    return parameters

def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def main():
    # Get user-specific parameters
    params = get_user_parameters()

    desired_width = params['desired_width']
    fps = params['fps']
    frame_skip = params['frame_skip']
    channel_width = params['channel_width']
    pixel_distance = params['pixel_distance']
    tracker_type = params['tracker_type']
    start_time = params['start_time']
    end_time = params['end_time']
    output_directory = params['output_directory']
    filename = params['filename']

    # Create a Tkinter root window and hide it
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select the video file
    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video files", "*.mov *.mp4 *.avi *.mkv")])
    if not video_path:
        print("No video file selected. Exiting...")
        return
    
    images_folder = os.path.join(output_directory, f"{filename}_images")
    labels_folder = os.path.join(output_directory, f"{filename}_labels")
    os.makedirs(images_folder, exist_ok=True)
    os.makedirs(labels_folder, exist_ok=True)

    csv_file_path = os.path.join(output_directory, f"{filename}.csv")

    # Read the video file
    video = cv2.VideoCapture(video_path)

    # Get the original width and height of the video
    original_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate the new height based on the desired width
    new_width = desired_width
    aspect_ratio = original_width / original_height
    new_height = int(new_width / aspect_ratio)

    # Initialize object tracker
    tracker = create_object_tracker(tracker_type)

    # Convert start and end times to frame numbers
    start_frame_number = int(start_time * fps)
    end_frame_number = int(end_time * fps)

    # Set the video capture to start from the specified start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_number)

    # Read the first three frames
    success, prev2_frame = video.read()
    success, previous_frame = video.read()

    i = start_frame_number + 1
    while i < start_frame_number + 10:
        success, current_frame = video.read()
        i += 1

    # Resize the frames to the new dimensions
    prev2_frame = cv2.resize(prev2_frame, (new_width, new_height))
    previous_frame = cv2.resize(previous_frame, (new_width, new_height))
    current_frame = cv2.resize(current_frame, (new_width, new_height))

    display_frame = current_frame.copy()

    # Calculate the double difference frame
    diff_frame = double_difference(previous_frame, current_frame, prev2_frame)

    # Select a region of interest (ROI) to track
    bbox = cv2.selectROI("Select Object to Track", diff_frame, fromCenter=False, showCrosshair=True)
    tracker.init(diff_frame, bbox)

    init_bbox_width, init_bbox_height = bbox[2], bbox[3]

    frame_counter = start_frame_number  # Initialize frame counter
    total_frames = start_frame_number

    x_prev = 0
    y_prev = 0

    time = start_time

    velocities = []
    times = []

    # Main loop to process each frame of the video
    while True:
        success, current_frame = video.read()  # Read the current frame
        if not success or frame_counter > end_frame_number:  # Stop if end of video or end time reached
            break

        total_frames += 1

        frame_counter += 1  # Increment frame counter

        if frame_counter % frame_skip != 0:  # Process every x frame
            continue

        # Resize the frame to the new dimensions
        current_frame = cv2.resize(current_frame, (new_width, new_height))

        display_frame = current_frame.copy()

        # Calculate the double difference frame
        diff_frame = double_difference(previous_frame, current_frame, prev2_frame)

        # Update the object tracker with the binary difference frame
        success, new_position = tracker.update(diff_frame)

        display_frame = current_frame.copy()

        frame_filename = f"frame_{frame_counter}.jpg"
        cv2.imwrite(os.path.join(images_folder, frame_filename), display_frame)

        # Use the initial bounding box coordinates and size to draw the ROI on the display frame
        if success:
            x, y = [int(val) for val in new_position[:2]]  # Only update the position, not the size
            cv2.rectangle(display_frame, (x, y), (x + init_bbox_width, y + init_bbox_height), (255, 0, 0), 2)

            center_x = x + init_bbox_width // 2
            center_y = y + init_bbox_height // 2

            # Draw a circle at the center of the bounding box
            cv2.circle(display_frame, (center_x, center_y), 5, (0, 0, 255), -1)

            x_diff = abs(abs(center_x) - abs(x_prev))
            y_diff = abs(abs(center_y) - abs(y_prev))
            distance = math.sqrt(x_diff**2 + y_diff**2) * pixel_distance

            x_prev = center_x
            y_prev = center_y

            velocity = distance / (frame_skip/fps)
            time = time + (frame_skip/fps)

            velocities.append(velocity)
            times.append(time)

            print(velocity)

            # Write bounding box information to a .txt file (YOLO format)
            bbox_txt_filename = os.path.join(labels_folder, f"frame_{frame_counter}.txt")
            with open(bbox_txt_filename, 'w') as f:
                # YOLOv5 format: class_id center_x center_y width height
                # Normalize the values by the frame width and height
                class_id = 0  # Assuming single class object detection; adjust if you have multiple classes
                center_x_norm = center_x / new_width
                center_y_norm = center_y / new_height
                width_norm = init_bbox_width / new_width
                height_norm = init_bbox_height / new_height
                f.write(f"{class_id} {center_x_norm} {center_y_norm} {width_norm} {height_norm}\n")

        # Display the result
        cv2.imshow("Object Tracking", display_frame)

        # Store the current frame as the previous frame for the next iteration
        prev2_frame = previous_frame.copy()
        previous_frame = current_frame.copy()

        # Press 'q' to quit the video
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Plot the velocity over time
    plt.plot(times[1:], velocities[1:], label='Velocity')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Velocity (mm/s)')
    plt.legend()
    plt.grid()
    plt.show()

    # Write velocity data to the CSV file
    with open(csv_file_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Time (seconds)', 'Velocity (mm/s)'])
        for t, v in zip(times[1:], velocities[1:]):
            csv_writer.writerow([t, v])

    print("Velocity and Time data saved to CSV file:", csv_file_path)

if __name__ == "__main__":
    main()

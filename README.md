# Face_Mask_Detector
This project is a real-time face mask detector built with Python, OpenCV, and a pre-trained YOLOv3 (You Only Look Once) model. The application uses a live webcam feed to identify people and classify them as wearing a mask, not wearing a mask, or wearing a mask incorrectly.
# Real-time Face Mask Detector with YOLOv3

A Python application that uses OpenCV and a pre-trained YOLOv3 deep learning model to detect face masks in a real-time video stream from a webcam.

## Features

-   **Real-time Detection:** Processes live video from a webcam to detect faces.
-   **Three-Class Classification:** Classifies detections into three categories: "Mask", "No Mask", and "Mask Worn Incorrectly".
-   **Visual Feedback:** Draws colored bounding boxes around detected faces (Green for Mask, Red for No Mask, Blue for Incorrect).
-   **High Performance:** Leverages the speed and accuracy of the YOLOv3 object detection algorithm.

## Technology Stack

-   **Python**
-   **OpenCV (`opencv-python`):** For video capture and image processing.
-   **NumPy:** For numerical operations and array manipulation.
-   **YOLOv3:** A pre-trained deep learning model for object detection.

## Setup and Usage

This project requires several pre-trained model files to be downloaded. Please follow these steps carefully.

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Anjali-Ajesh/face-mask-detector.git](https://github.com/Anjali-Ajesh/face-mask-detector.git)
    cd face-mask-detector
    ```

2.  **Install Dependencies:**
    It's recommended to use a virtual environment.
    ```bash
    # Create and activate a virtual environment
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

    # Install the required libraries
    pip install opencv-python numpy
    ```

3.  **Download Model Files:**
    You need to download three files for the YOLO model. Create a `model` folder in your project directory and place them inside it.

    * **`yolov3-wider_16000.weights`:** The pre-trained YOLOv3 model weights.
        * [Download Link](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/yolov3-wider_16000.weights) (Click "Download" on the file page).
    * **`yolov3-wider.cfg`:** The YOLOv3 model configuration file.
        * [Download Link](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/yolov3-wider.cfg) (Right-click the "Raw" button and "Save Link As...").
    * **`obj.names`:** The file containing the names of the classes the model can detect.
        * [Download Link](https://github.com/chandrikadeb7/Face-Mask-Detection/blob/master/obj.names) (Right-click the "Raw" button and "Save Link As...").

    Your project structure should look like this:
    ```
    face-mask-detector/
    ├── model/
    │   ├── yolov3-wider_16000.weights
    │   ├── yolov3-wider.cfg
    │   └── obj.names
    └── face_mask_detector.py
    ```

4.  **Run the Detector:**
    Execute the Python script from your terminal. Make sure your webcam is connected and enabled.
    ```bash
    python face_mask_detector.py
    ```
    A window will appear showing your webcam feed with the detections. Press the **'q'** key to quit.

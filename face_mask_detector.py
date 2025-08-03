# face_mask_detector.py

import cv2
import numpy as np

def run_mask_detector():
    """
    Initializes the webcam and runs the YOLOv3 model for face mask detection.
    """
    # --- 1. Load YOLO Model ---
    try:
        # Load the network using the pre-trained weights and configuration file
        net = cv2.dnn.readNet("model/yolov3-wider_16000.weights", "model/yolov3-wider.cfg")
    except cv2.error as e:
        print("Error loading model files. Make sure 'yolov3-wider_16000.weights' and 'yolov3-wider.cfg' are in the 'model' directory.")
        return

    # Load the class names
    try:
        with open("model/obj.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        print("Error: 'obj.names' not found in the 'model' directory.")
        return
        
    # Get the names of the output layers
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    # Define colors for the classes (Mask, No Mask, Incorrect)
    colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)] 

    # --- 2. Initialize Webcam ---
    cap = cv2.VideoCapture(0) # 0 is the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam feed. Press 'q' to quit.")

    while True:
        # --- 3. Read Frames from Webcam ---
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width, channels = frame.shape

        # --- 4. Prepare Frame for YOLO ---
        # Create a blob from the image and perform a forward pass
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # --- 5. Process Detections ---
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # Filter out weak detections
                if confidence > 0.5:
                    # Object detected, calculate bounding box coordinates
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Apply Non-Max Suppression to remove overlapping boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # --- 6. Draw Bounding Boxes on the Frame ---
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                confidence = confidences[i]
                
                # Draw rectangle and text
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = f"{label} {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- 7. Display the Result ---
        cv2.imshow("Face Mask Detector", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # --- 8. Cleanup ---
    print("Closing application...")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_mask_detector()

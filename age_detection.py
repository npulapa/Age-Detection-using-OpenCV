import cv2
import numpy as np

# ----------- Load Models -----------

# Face detection model
face_model = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)

# Age detection model
age_model = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt",
    "models/age_net.caffemodel"
)

# Age categories
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)',
            '(25-32)', '(38-43)', '(48-53)', '(60-100)']


# ----------- Start Webcam -----------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()


# ----------- Processing Loop -----------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]

    # Convert image to blob (for face detection)
    blob = cv2.dnn.blobFromImage(
        frame, 1.0, (300, 300),
        (104, 177, 123)
    )

    face_model.setInput(blob)
    detections = face_model.forward()

    # Loop through detections
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.7:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            # Ensure coordinates are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            # Prepare face for age prediction
            face_blob = cv2.dnn.blobFromImage(
                face, 1.0, (227, 227),
                (78.426, 87.768, 114.895),
                swapRB=False
            )

            age_model.setInput(face_blob)
            preds = age_model.forward()
            age = AGE_LIST[preds[0].argmax()]

            # ----------- Draw Output -----------

            label = f"Age: {age}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # Show output window
    cv2.imshow("Age Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break


# ----------- Release Resources -----------

cap.release()
cv2.destroyAllWindows()

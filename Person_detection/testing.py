import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# load pre-trained model
model = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2")

# open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # resize and normalize
    img = cv2.resize(frame, (320,320))
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.uint8)

    # run detection
    results = model(img)
    boxes = results["detection_boxes"][0].numpy()
    scores = results["detection_scores"][0].numpy()
    classes = results["detection_classes"][0].numpy().astype(int)

    # loop through detections
    h, w, _ = frame.shape
    for i in range(len(boxes)):
        if scores[i] > 0.6 and classes[i] == 1:   # 1 = person
            ymin, xmin, ymax, xmax = boxes[i]
            x1, y1, x2, y2 = int(xmin*w), int(ymin*h), int(xmax*w), int(ymax*h)
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

    cv2.imshow("tensorflow camera", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np

# net = cv2.dnn.readNetFromONNX("model.onnx") # shallow cnn
# net = cv2.dnn.readNetFromONNX("resnet18.onnx")
# net = cv2.dnn.readNetFromONNX("resnet50.onnx")
net = cv2.dnn.readNetFromONNX("resnet152.onnx")
# net = cv2.dnn.readNetFromONNX("vit_b_16.onnx")


cap = cv2.VideoCapture(0)

classes = ["palm", "L", "fist", "fist_sideways", "thumb", "index", "ok", "palm_sideways", "C", "claw"]
while True:
    ret, frame = cap.read()
    # input_frame = cv2.resize(frame, (132, 132)) # shallow cnn
    input_frame = cv2.resize(frame, (224, 224)) #resnet

    lowerHSV = np.array([0, 17, 0])
    upperHSV = np.array([44, 170, 241])
    hsv = cv2.cvtColor(input_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lowerHSV, upperHSV)

    segmented = cv2.bitwise_and(input_frame, input_frame, mask=mask)
    cv2.imshow("segmented", segmented)

    input_gray = cv2.cvtColor(segmented, cv2.COLOR_BGR2GRAY)
    input_gray = cv2.addWeighted(input_gray, 2.3, np.zeros(input_gray.shape, input_gray.dtype), 0, 30)
    input_gray_3channel = cv2.cvtColor(input_gray, cv2.COLOR_GRAY2RGB)
    blob = cv2.dnn.blobFromImage(input_gray_3channel, 1.0 / 255, (224, 224), (0, 0), crop=False)
    # print(np.array(blob).shape)
    cv2.imshow("gray", input_gray_3channel)

    net.setInput(blob)
    preds = net.forward()
    biggest_pred_index = np.array(preds)[0].argmax()
    confidence = np.array(preds)[0].max()
    # print(preds)
    # print(biggest_pred_index)
    # print("Predicted class:", classes[biggest_pred_index])
    if biggest_pred_index < 10:
        cv2.putText(frame, classes[biggest_pred_index] + " " + str(round(confidence, 2)) + "%", (50, 50), cv2.FONT_HERSHEY_SIMPLEX ,
                1, (240, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow("frame", frame)

    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

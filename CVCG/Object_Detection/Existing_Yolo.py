import cv2
import numpy as np

class Existing_Yolo:
    def object_segmentation(self, spath, dpath):

        def get_output_layers(net):
            layer_names = net.getLayerNames()
            try:
                output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
            except:
                output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

            return output_layers

        def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
            label = str(classes[class_id])

            color = COLORS[class_id]

            cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 5)

            # cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        image = cv2.imread(spath)

        Width = image.shape[1]
        Height = image.shape[0]
        scale = 0.00392

        classes = None

        with open("..\\Files\\coco.names", 'r') as f:
            classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet("..\\Files\\yolov8.weights", "..\\Files\\yolov8.cfg")

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.5
        nms_threshold = 0.4


        for out in outs:

            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])


        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        ROI_number = 0
        global x1
        global y1
        global w1
        global h1
        conf_val = []
        for i in indices:
            temp = []
            box = boxes[i]
            x = box[0]
            x1 = x
            y = box[1]
            y1 = y
            w = box[2]
            w1 = w
            h = box[3]
            h1 = h
            draw_prediction(image, class_ids[i], confidences[i], round(x1), round(y1), round(x1 + w1), round(y1 + h1))
            temp.append(confidences[i])
            cv2.imwrite(dpath, image)
            conf_val.append(temp)

        return conf_val
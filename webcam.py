import cv2
import numpy as np
# import time


classes = None
with open("./config/coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes)))


def get_output_layers(net):
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[200-1],layer_names[227-1],layer_names[254-1]]
    return output_layers
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

HEIGHT = 712
WIDTH = 712
SCALE = 0.00392
net = cv2.dnn.readNet("./config/yolov3.weights", "./config/yolov3.cfg")
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    # cv2.imshow("output",frame)
    try:
        
        image = cv2.resize(frame,(HEIGHT,WIDTH))
        blob = cv2.dnn.blobFromImage(image, SCALE, (416,416), (0,0,0), True, crop=False)
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
                    center_x = int(detection[0] * WIDTH)
                    center_y = int(detection[1] * HEIGHT)
                    w = int(detection[2] * WIDTH)
                    h = int(detection[3] * HEIGHT)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])
        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        for i in indices:
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
    
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
        cv2.imshow("output",image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except Exception as e:
        pass
        # print(e)
cap.release()
cv2.destroyAllWindows()
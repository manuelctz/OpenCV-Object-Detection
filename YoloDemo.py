import cv2
import numpy as np

### Webcam Setup ###
cap = cv2.VideoCapture(0)
widthT = 320
heightT = 320 #size of blob image 320 because using yolo 320
confThreshold = 0.45

### SAVE VIDEO ###
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('VideoTest.mp4',fourcc,20.0,(640,480))

### Names of the Yolo Classes ###
CocoFile = 'coco.names'
CocoNames = []
with open(CocoFile,'rt') as f:
    CocoNames = f.read().rstrip('\n').split('\n')
print(CocoNames)
print(len(CocoNames))

### Setting Yolo and loading the network ###
ModelStructure = '/home/manu/OpenCV_Object_Detection/yolov3-tiny.cfg'
ModelWeights =  '/home/manu/OpenCV_Object_Detection/yolov3-tiny.weights'

net = cv2.dnn.readNetFromDarknet(ModelStructure,ModelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
#Function to find the objects listed in the coco library
def findObjects(outputs,img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []

    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))
    print(len(bbox))
    indices = cv2.dnn.NMSBoxes(bbox,confs,confThreshold,nms_threshold= 0.3)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x,y,w,h = box[0],box[1],box[2],box[3]
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
        cv2.putText(img,f'{CocoNames[classIds[i]].upper()} {int(confs[i]*100)}%',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,(255,0,255),2)

### Calling the webcam to get video ###
while True:
    success, img = cap.read()
    img = cv2.flip(img, 180)
    blob = cv2.dnn.blobFromImage(img,1/255,(widthT,heightT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    #print(layerNames)
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print(outputNames)
    #print(net.getUnconnectedOutLayers())
    outputs = net.forward(outputNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)

    findObjects(outputs,img)
    #save the video as mp4 file
    out.write(img)
    cv2.imshow("Video",img)
    #Finish the program with the q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

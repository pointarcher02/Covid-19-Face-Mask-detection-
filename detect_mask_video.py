#first we import the necessary packages and libraries
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import time
import cv2
import os

def detect_and_predict(frame,faceNet,maskNet):
    # so we are passing three things that is frame, and pretrained models faceNet and maskNet
    # faceNet is for detecting the face
    #maskNet for detecting the mask:- the model we trained

    (h,w)=frame.shape[:2]
    #constructing a blob from the dimensions of the frame
    blob=cv2.dnn.blobFromImage(frame,1.0,(224,224),(104.0,177.0,123.0))

    #now blob is 4dimensional being fed into the network
    #passing and doing face predictions
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    #now initializing the list of faces , their corresponding locations and face mask predictions
    faces=[]
    locs=[]
    preds=[]

    for i in range(0,detections.shape[2]):
        #initializing the confidience or probability for each detection
        confidience=detections[0,0,i,2]

        #now filter out the weak detections by ensuring a threshold for the confidence
        if confidience>0.5:
            #compute the bounding box for the face
            box=detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            #now ensuring that the bounding box falls within the dimension of the frame
            (startX,startY)=(max(0,startX),max(0,startY))
            (endX,endY)=(min(w-1,endX),min(h-1,endY))

            #now finding the ROI of the face and applying preprocessing of the frame
            face=frame[startY:endY, startX:endX]
            face=cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
            face=cv2.resize(face,(224,224))
            face=img_to_array(face)
            face = preprocess_input(face)

            faces.append(face)
            locs.append((startX,startY,endX,endY))

    
    if(len(faces)>0):
        #now we predict whether there is a mask or not for all the faces in the form of batches
        #instead of going for one by one

        faces=np.array(faces,dtype="float32")
        preds=maskNet.predict(faces,batch_size=32)

    return (locs,preds)

#loading our serialized face detector model from the disk
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

#loading the mask detector model which is already trained by us
maskNet=load_model("mask_detector.model")

#now starting the video stream
vs=VideoStream(src=0).start()

while(True):

    #extract the frame from the videostream and resize it along width
    frame=vs.read()
    frame=imutils.resize(frame,width=400)

    #now check the frame for face and mask predictions
    (locs,preds)=detect_and_predict(frame,faceNet,maskNet)

    #now loop over the face detections and corresponding predictions
    for (box,pred) in zip(locs,preds):
        #unpacking the material got from function
        (startX,startY,endX,endY)=box
        (mask,without_mask)=pred

        label="Mask" if mask>without_mask else "No mask"
        color=(0,255,0) if label=="Mask" else (0,0,255)

        #also include the probability
        label="{}: {:.2f}%".format(label, max(mask, without_mask) * 100)

        cv2.putText(frame,label,(startX,startY-10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame,(startX, startY), (endX, endY), color, 2)

    
    cv2.imshow("Frame",frame)
    key=cv2.waitKey(1) & 0xFF


    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()















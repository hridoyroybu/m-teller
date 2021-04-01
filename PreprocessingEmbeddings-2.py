from imutils import paths
import numpy as np
import imutils
import pickle
import cv2
import os

dataset = "dataset"
embeddingFile = "output/embeddings.pickle"
embeddingModel = "openface.nn4.small2.v1.t7"

#initialization of caffe model for face detection
prototxt = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"

#loading caffe model for face detection
#detection face from Image via Caffe deep learning
detector = cv2.dnn.readNetFromCaffe(prototxt , model)

#loading pytorch model file for extract facial embeddings
#extracting facial embedding vai deep learning feature extraction
embedder = cv2.dnn.readNetFromTorch(embeddingModel)

#getting image path
imagePaths = list(paths.list_images(dataset))

#initialization
knownEmbeddings = []
knownNames = []
total = 0
conf = 0.5

for (i,imagePath) in enumerate(imagePaths):
    print("Processing image{}/{}".format(i+1 , len(imagePaths)))
    name = imagePath.split(os.path.sep)[-2]
    image = cv2.imread(imagePath)
    image = imutils.resize(image , width=600)
    (h,w) = image.shape[:2]

    #converting image to blob for dnn face detection
    imageBlob = cv2.dnn.blobFromImage(
        cv2.resize(image,(300,300)),1.0,(300,300),(104.0, 177.0 , 123.0))

    #setting inout blob image
    detector.setInput(imageBlob)
    #detection the face
    detections = detector.forward()

    if len(detections) > 0:
        i = np.argmax(detections[0,0,:,2])
        confidence = detections[0,0,i,2]

        if confidence > conf:
            #ROI range of interest
            box = detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX , startY , endX , endY) = box.astype("int")
            face = image[startY:endY , startX:endX]
            (fH , fW) = face.shape[:2]
            if fW < 20 or fH < 20:
                continue
            faceBlob = cv2.dnn.blobFromImage(face,1.0/255,(96,96),(0,0,0),swapRB=True, crop=False)
            embedder.setInput(faceBlob)
            vec = embedder.forward()
            knownNames.append(name)
            knownEmbeddings.append(vec.flatten())
            total += 1

print("Embedding:{0} ".format(total))
data = {"embeddings": knownEmbeddings, "names":knownNames}
f  = open(embeddingFile, "wb")
f.write(pickle.dumps(data))
f.close()
print("Process Completed")













    

#importing the necessary packages 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

#initializing the initial learning rate , epochs and batch size
INIT_LR=1e-4
EPOCHS=20
BS=32

DIRECTORY=r"C:\Users\shash\Desktop\Facemask_detection _project\data"
CATEGORY=["with_mask", "without_mask"]


# grab the list of images in our dataset directory, then initialize the list of data (i.e., images) and class images
print("[INFO] loading images...")

data=[]
labels=[]

for category in CATEGORY:
    path=os.path.join(DIRECTORY,category)
    for img in os.listdir(path):
        img_path=os.path.join(path,img)
        image=load_img(img_path,target_size=(224,224))
        image=img_to_array(image)
        image=preprocess_input(image)
        
        data.append(image)
        labels.append(category)
    

#now we are one hot encoding the labels 
lb=LabelBinarizer()
labels=lb.fit_transform(labels)
labels=to_categorical(labels)

#converting the lists into numpy arrays
data=np.array(data, dtype="float32")
labels=np.array(labels)

trainX,testX,trainY,testY=train_test_split(data,labels,test_size=0.20, stratify=labels, random_state=42)

#constructing the training image generator for data augmentation
aug=ImageDataGenerator(
    rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

#loading the MobileNetV2 network as the base model and later the head model
# will be designed manually 

#we defining the base model as MobileNetv2 with weights trained on Imagenet dataset
basemodel= MobileNetV2(weights="imagenet",include_top=False,input_tensor=Input(shape=(224, 224, 3)))

#now we construct the head of the model which will be placed on top of the base model
headmodel=basemodel.output
headmodel=AveragePooling2D(pool_size=(7,7))(headmodel)
headmodel=Flatten(name="flatten")(headmodel)
headmodel=Dense(128, activation="relu")(headmodel)
headmodel=Dropout(0.5)(headmodel)
headmodel=Dense(2,activation="softmax")(headmodel)

#now to clearly define the pipeline we need to make sure that we place the head model on the top of basemodel
model=Model(inputs=basemodel.input,outputs=headmodel)

# now if we freeze the baselayer then the weights will not be changed of baselayer while backpropogation
#only head model which we have constructed will be changed

for layer in basemodel.layers:
    layer.trainable=False

#compiling the model
print("[INFO] compiling model...")
opt = Adam(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#now training the head of the network
H=model.fit(
    aug.flow(trainX,trainY,batch_size=BS),
    steps_per_epoch=len(trainX) // BS,
    validation_data=(testX,testY),
    validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs =model.predict(testX,batch_size=BS)

#now we need to find the label as probability is always between 0 and 1 for each image prediction
predIdxs=np.argmax(predIdxs,axis=1)

#show a classification report
print(classification_report(testY.argmax(axis=1),predIdxs,target_names=lb.classes_))

# serialize the model to disk
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

#now inorder for the visualization we plot the loss and accuracy
N=EPOCHS

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,N),H.history["loss"],label="trainloss")
plt.plot(np.arange(0,N),H.history["val_loss"],label="val_loss")
plt.plot(np.arange(0,N),H.history["accuracy"],label="train_acc")
plt.plot(np.arange(0,N),H.history["val_accuracy"],label="val_acc")
plt.title("Training loss and accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")











    




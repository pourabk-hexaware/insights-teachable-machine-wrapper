import cv2
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
import time
import math
from PIL import Image, ImageOps

video_capture = cv2.VideoCapture(0)

video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 430)

fps = 0
frame_num = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = os.path.join(os.getcwd(), 'models', 'keras_model.h5')

sess = tf.Session()

graph = tf.compat.v1.get_default_graph()
set_session(sess)
model = load_model(MODEL_PATH)

# Fetch labels
x = None
labels = list()
with open(os.path.join(os.getcwd(), 'models', 'labels.txt'), 'r') as f:
    x = [line.rstrip('\n') for line in f]

for item in x:
    print("Item : {}".format(item))
    split_label = item.split(" ")
    label_index = split_label[0]
    merger = ""
    for i in range(1, len(split_label)):
        merger += split_label[i] + " "

    print(("Merger Index : {} ".format(merger)))
    labels.append({"id": label_index, "label": merger[0:len(merger)-1]})

while(True):
        start_time = time.time()
        # vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
        ret, frame = video_capture.read()
        if not ret:
            break
        # Display the resulting frame
        image = frame[:,:,0:3]
        image=Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

        #resize the image to a 224x224 with the same strategy as in TM2:
        #resizing the image to be at least 224x224 and then cropping from the center
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.ANTIALIAS)
        image = image.convert('RGB')

        #turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        #load the labels
       
        print(labels[0])
        # run the inference
        prediction = model.predict(data)
     
     

        for i in range(0, len(prediction[0])):
            if(prediction[0][i]>0.70):
                print("Id : {}".format(i))
                print("Score: {}".format(float(prediction[0][i])))
                print("label: {}".format(labels[i]['label']))
                prediction_info = 'Predicted: {} | Score {}'.format(labels[i]['label'], prediction[0][i])
                cv2.putText(frame, prediction_info, (10, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        end_time = time.time()
        fps = fps * 0.9 + 1/(end_time - start_time) * 0.1
        start_time = end_time
        frame_info = 'Frame: {0} | FPS: {1:.2f}'.format(frame_num, fps)
       
        cv2.putText(frame, frame_info, (10, frame.shape[0]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()

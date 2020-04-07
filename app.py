from tensorflow.keras.models import load_model
from tensorflow.python.keras.backend import set_session
import tensorflow as tf
from flask import Flask, request, render_template, jsonify, send_file, url_for
import os
from PIL import Image, ImageOps
import numpy as np
import math
import time
import base64

app = Flask(__name__)
app.static_folder = 'static'
dir_path = os.path.dirname(os.path.realpath(__file__))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

MODEL_PATH = os.path.join(os.getcwd(), 'models', 'keras_model.h5')

sess = tf.Session()

graph = tf.compat.v1.get_default_graph()
set_session(sess)
model = load_model(MODEL_PATH)

def read_labels():
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

    return labels


@app.route('/detect', methods=['POST'])
def post_example():

    global sess
    global graph
    with graph.as_default():
        
        # perform the prediction
        set_session(sess)
        np.set_printoptions(suppress=True)

        if not request.headers.get('Content-type') is None:
            if(request.headers.get('Content-type').split(';')[0] == 'multipart/form-data'):
                if 'image' in request.files.keys():
                    print("inside get image statement")
                    file = request.files['image']
                    img = Image.open(file.stream)  # PIL image
                    uploaded_img_path = os.path.join(os.getcwd(), 'static', 'uploads', file.filename)
                    print("Upload Path : {}".format(uploaded_img_path))

                    img.save(uploaded_img_path)

                    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                   
                    print("Image path {}".format(uploaded_img_path))
                    image = Image.open(uploaded_img_path)

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
                    labels = read_labels()
                    print(labels[0])
                    # run the inference
                    prediction = model.predict(data)
                    print(prediction[0])

                    scores = list()
                    for i in range(0, len(prediction[0])):
                        print("Id : {}".format(i))
                        print("Score: {}".format(float(prediction[0][i])))
                        print("label: {}".format(labels[i]['label']))
                        scores.append({"id": i, "label": float(prediction[0][i]), "score": labels[i]['label']})
                   
                    result = {
                        "inference": scores
                    }

                    return jsonify(result), 200
                          
                else:
                    return jsonify(get_status_code("Invalid body", "Please provide valid format for Image 2")), 415

            elif(request.headers.get('Content-type') == 'application/json'):
                if(request.data == b''):
                    return jsonify(get_status_code("Invalid body", "Please provide valid format for Image")), 415
                else:
                    body = request.get_json()
                    if "image_string" in body.keys():
                        str_image = body['image_string']
                        # str_image = img_string.split(',')[1]
                        imgdata = base64.b64decode(str_image)
                        uploaded_img_path = os.path.join(os.getcwd(), 'static', 'uploads', str(time.time())+".jpg")
                        # img = "uploads\\" +  str(int(round(time.time() * 1000))) + "image_file.jpg"
                        with open(uploaded_img_path, 'wb') as f:
                            f.write(imgdata)

                        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
                   
                        print("Image path {}".format(uploaded_img_path))
                        image = Image.open(uploaded_img_path)

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
                        labels = read_labels()
                        print(labels[0])
                        # run the inference
                        prediction = model.predict(data)
                        print(prediction[0])

                        scores = list()
                        for i in range(0, len(prediction[0])):
                            scores.append({"id": i, "label": float(prediction[0][i]), "score": labels[i]['label']})
                    
                        result = {
                            "inference": scores
                        }
                        return jsonify(result), 200

            else:
                return jsonify(get_status_code("Invalid header", "Please provide correct header with correct data")), 415

        else:
            return jsonify(get_status_code("Invalid Header", "Please provide valid header")), 401


def get_status_code(argument, message):
    res = {
        "error": {
            "code": argument,
            "message": message
        }
    }
    return res

if __name__=="__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host="0.0.0.0", port=port)

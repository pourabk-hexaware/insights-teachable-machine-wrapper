# Inisghts Demo Code - Python Wrapper for Google Teachable Machine
##### A python based backend for hosting ML models from Google's Teachable Machine and Running demos on webcam
&nbsp;

This codebase does the following steps:

  - Using exported Keras models as REST APIs
  - Running exported models at runtime using webcam


## Inspiration
This project is greatly inspired from AutoML implementations

### Technology

This project uses a number of open source projects to work properly:

* [Python] - awesome language we love

### Environment Setup



#### Installation

This project requires the standard [Python](https://www.python.org/) 3.6+ to run `(Use 3.6 only, not 3.7 or 3.8)`

```sh
$ git clone git@github.com:pourabk-hexaware/teachable-machine-wrapper.git
```
OR

```sh
$ https://github.com/pourabk-hexaware/teachable-machine-wrapper.git
```

```sh
$ cd teachable-machine-wrapper
$ pip install -r requirements.txt
```
#### Paste download Keras `keras_model.h5 and label.txt` files under`model` folder
#### To run webcam demo

```sh
$ python webcam.py
```
#### To Run Web server
```sh
$ python app.py
```

#### REST endpoints 
###### `http://localhost:5000/detect` for for form-data uploads with key `'image'` or,
###### `base64` input { "image_string": "/9....." } as body using header `"application/json"`





License
----

Public


 
   [Python]: <https://www.python.org/>
 

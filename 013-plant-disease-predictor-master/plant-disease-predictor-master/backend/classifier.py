from keras.preprocessing import image
import numpy as np
from backend.deeplearning import graph,output_list
from keras.models import load_model


model = load_model('backend/models/model1.hdf5')

def classify(image_data='', model_path=''):
    global model

    if not model_path.endswith('model1.hdf5'):
        model = load_model(model_path)

    img = image.img_to_array(image_data)
    img = np.expand_dims(img, axis=0)
    img = img/255
    with graph.as_default():
        prediction = model.predict(img)

    prediction_flatten = prediction.flatten()
    max_val_index = np.argmax(prediction_flatten)
    plant_info = output_list[max_val_index]
    result = "Plant:{0}\nHealth Status:{1}".format(plant_info[0], plant_info[1])
    
    return result

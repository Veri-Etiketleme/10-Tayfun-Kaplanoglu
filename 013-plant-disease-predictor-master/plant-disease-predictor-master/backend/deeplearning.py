import tensorflow as tf

graph = tf.get_default_graph()

HEALTH_STATUS={
    'c_0': ['Apple', 'Apple scab'], 
    'c_1': ['Apple', 'Black rot'], 
    'c_2': ['Apple', 'Cedar apple rust'], 
    'c_3': ['Apple', 'healthy'], 
    'c_4': ['Blueberry', 'healthy'], 
    'c_5': ['Cherry_(including_sour)', 'Powdery mildew'], 
    'c_6': ['Cherry_(including_sour)', 'healthy'], 
    'c_7': ['Corn_(maize)', 'Cercospora leaf spot Gray leaf spot'], 
    'c_8': ['Corn_(maize)', 'Common rust '], 
    'c_9': ['Corn_(maize)', 'Northern Leaf Blight'], 
    'c_10': ['Corn_(maize)', 'healthy'], 
    'c_11': ['Grape', 'Black rot'], 
    'c_12': ['Grape', 'Esca (Black Measles)'], 
    'c_13': ['Grape', 'Leaf blight (Isariopsis Leaf Spot)'], 
    'c_14': ['Grape', 'healthy'], 
    'c_15': ['Orange', 'Haunglongbing (Citrus greening)'], 
    'c_16': ['Peach', 'Bacterial spot'], 
    'c_17': ['Peach', 'healthy'], 
    'c_18': ['Pepper bell', 'Bacterial spot'], 
    'c_19': ['Pepper bell', 'healthy'], 
    'c_20': ['Potato', 'Early blight'], 
    'c_21': ['Potato', 'Late blight'], 
    'c_22': ['Potato', 'healthy'], 
    'c_23': ['Raspberry', 'healthy'], 
    'c_24': ['Soybean', 'healthy'], 
    'c_25': ['Squash', 'Powdery mildew'], 
    'c_26': ['Strawberry', 'Leaf scorch'], 
    'c_27': ['Strawberry', 'healthy'], 
    'c_28': ['Tomato', 'Bacterial spot'], 
    'c_29': ['Tomato', 'Early blight'], 
    'c_30': ['Tomato', 'Late blight'], 
    'c_31': ['Tomato', 'Leaf Mold'], 
    'c_32': ['Tomato', 'Septoria leaf spot'], 
    'c_33': ['Tomato', 'Spider mites Two-spotted spider mite'], 
    'c_34': ['Tomato', 'Target Spot'], 
    'c_35': ['Tomato', 'Tomato Yellow Leaf Curl Virus'], 
    'c_36': ['Tomato', 'Tomato mosaic virus'], 
    'c_37': ['Tomato', 'healthy']
}

output_list = list(HEALTH_STATUS.values())

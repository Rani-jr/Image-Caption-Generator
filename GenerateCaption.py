import numpy as np

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import load_model

from pickle import load
from PIL import Image
from IPython.display import display 
import argparse

import warnings
warnings.filterwarnings("ignore")

def argParser():
    
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("-i", "--image", required=True, help="Path to Image File ")
    args = vars(arg_parse.parse_args())
    args = args['image']
    
    return args

def extract_features(args, model):

    image = Image.open(argParser())
    image = image.resize((299,299))
    image = np.array(image)
    #if the image has 4 channels, we convert them into 3 channels
    if image.shape[2] == 4:
        image = image[..., :3]
        
    image = np.expand_dims(image, axis=0)
    image = image/127.5
    image = image - 1.0
    feature = model.predict(image)
    
    return feature

def word_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    
    return None

def generate_caption(model, tokenizer, image, max_length):
    cap = '<start> '
    for i in range(max_length):
        seq = tokenizer.texts_to_sequences([cap])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        prediction = model.predict([image, seq], verbose=0)
        prediction = np.argmax(prediction)
        word = word_id(prediction, tokenizer)
        
        if word is None:
            break
        
        cap += ' ' + word
        if word == 'end':
            break
    
    return cap

max_length = 33
tokenizer = load(open("C:/ImageCaptionGenerator/tokenizer.p", "rb"))
model = load_model("C:/ImageCaptionGenerator/model_10.h5")
xception_model = Xception(include_top = False, pooling="avg")

if __name__ == "__main__":
    args = argParser()
    image = extract_features(args, xception_model)
    img = Image.open(args)

    caption = generate_caption(model, tokenizer, image, max_length)
    print('\n\n')
    print(caption[:-4])
    display(Image.open(args))
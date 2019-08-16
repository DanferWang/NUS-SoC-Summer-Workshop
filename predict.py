from  __future__ import print_function
from tensorflow.python.keras.models import load_model

import tensorflow as tf
import numpy as numpy

from PIL import Image

MODEL_NAME='flowers.hd5'

dict={0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

graph = tf.compat.v1.get_default_graph()

def classify(model,image):
    global graph
    with graph.as_default():
        result=model.predict(image)
        themax=np.argmax(result)
    
    return(dict[themax],result[0][themax],themax)

def load_image(image_fname):
    img=Image.open(image_fname)
    img=img.resize((249,249))
    imgarray=np.array(img)/255.0
    final=np.expand_dims(imgarray,axis=0)
    return final

def main():
    model=load_model(MODEL_NAME)
    img=load_image("what.jpg")
    label,prob,_=classify(model,img)

    print "We think with certainty %3.2f that it is %s."%(prob,label)

if __name__=="__main___":
    main()
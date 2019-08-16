from  __future__ import print_function
from tensorflow.python.keras.models import load_model

import tensorflow as tf
import numpy as np

from PIL import Image

MODEL_NAME='cat.hd5'

dict={0:'MaineCoon',1:'Ocelot',2:'Singapura',3:'TurkishVan'}

def classify(model,image):
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
    t_correct=0
    print('\n    Start testing', '\n')
    for i in range(70):
        img = load_image('cat_photos/02/what ({}).jpg'.format(i+1))
        label,prob,_=classify(model,img)
        if label==dict[3]:
            t_correct=t_correct+1
        print("We think with certainty %3.2f that it is %s."%(prob,label))
    total_acc=t_correct/70
    print("The total accuracy is %3.2f." % (total_acc))

if __name__=="__main__":
    main()

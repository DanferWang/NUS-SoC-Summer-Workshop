from  __future__ import print_function
from tensorflow.python.keras.models import load_model

import tensorflow as tf
import numpy as np

from PIL import Image

MODEL_NAME='rubbish.hd5'

dict={0:'cardboard',1:'glass',2:'metal',3:'paper',4:'plastic',5:'wall'}

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
    for k in range(6):
        print('\n    Start testing', dict[k], '\n')
        correct = 0
        for i in range(100):
            img = load_image('testing/{}/{}{}.jpg'.format(dict[k], dict[k], i+1))
            label,prob,_=classify(model,img)
            if label==dict[k]:
                correct=correct+1
                t_correct=t_correct+1
            print("We think with certainty %3.2f that it is %s."%(prob,label))
        class_acc=correct/100
        print('The accuracy of %s. class is %3.2f.'%(dict[k],class_acc))
    total_acc=t_correct/600
    print('The total accuracy is ',total_acc)

if __name__=="__main__":
    main()

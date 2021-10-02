import os
import numpy as np
from tensorflow import keras
from tensorflow.python.keras.applications.vgg19 import VGG19

from skimage.transform import resize
from keras.models import load_model

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
import cv2
from keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# This function prepares the training images. It take three parameters - path to the good and bad images folder, also
# model name
def doTraining(goodpath,badpath,modelname):
    im_width = 128
    im_height = 128
    thegoodpath =goodpath
    thebadpath = badpath
    good = len(next(os.walk(thegoodpath))[2])
    bad = len(next(os.walk(thebadpath))[2])
    totalimagecount = good + bad
    print(totalimagecount)

    X = np.zeros(((totalimagecount), im_height, im_width, 3), dtype=np.float)
    y = np.zeros(((totalimagecount)), dtype=np.float)

    goodimages = next(os.walk(thegoodpath))[2]
    badimages = next(os.walk(thebadpath))[2]
    thecounter = 0
    for goodimage, badimage in zip(goodimages,badimages):
        vv = goodpath+"//" + goodimage
        img = cv2.imread(vv)
        img = np.array(img)
        x_img =img
        x_img = resize(x_img, (im_width, im_width, 3), mode='constant', preserve_range=True)
        x_img = img_to_array(x_img)
        print(x_img.shape)
        X[thecounter] = x_img / 255

        y[thecounter] = 1.0

        thecounter =thecounter +1

        vv = badpath + "//" + badimage
        img = cv2.imread(vv)

        img = np.array(img)
        x_img = img
        x_img = resize(x_img, (im_width, im_width, 3), mode='constant', preserve_range=True)
        x_img = img_to_array(x_img)
        X[thecounter] = x_img / 255

        y[thecounter] = 0.0
        thecounter = thecounter + 1


    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.30, random_state=2019)

    X_train = np.append(X_train, [np.fliplr(i) for i in X_train], axis=0)
    y_train = np.append(y_train, [i for i in y_train], axis=0)

    X_train = np.append(X_train, [np.flipud(i) for i in X_train], axis=0)
    y_train = np.append(y_train, [i for i in y_train], axis=0)

    X_train = np.append(X_train, [np.fliplr(i) for i in X_train], axis=0)
    y_train = np.append(y_train, [i for i in y_train], axis=0)

    model = keras.models.Sequential()
    model.add(VGG19(include_top=False, weights='imagenet', input_shape=(128, 128, 3), pooling='max'))
    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    #model.add(Dense(30, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))


    model.compile(optimizer=Adam(), loss="binary_crossentropy", metrics=["accuracy"])
    themo = modelname + ".h5"
    model.summary()

    callbacks = [EarlyStopping(patience=10, verbose=1),
                 ReduceLROnPlateau(factor=0.15, patience=3, min_Ir=0.00001, verbose=1),
                 ModelCheckpoint(themo, verbose=1, save_best_only=True, save_weights_only=False)]

    model.fit(X_train, y_train, batch_size=5, epochs=100, callbacks=callbacks,
                        validation_data=(X_valid, y_valid))


# This function classifies a single image, it takes in the image to be classified and the saved model
def classifyimage(image,model):

    model = load_model(model)
    img = cv2.imread(image)
    resizedImage = cv2.resize(img, (128, 128))
    imagenp = np.array(resizedImage / 255)
    expanded = np.expand_dims(imagenp, axis=0)
    output = model.predict(expanded)
    if (output[0][0] < 0.5):
        theimage =  cv2.putText(cv2.resize(img, (256, 256)),"Bad Image",(50,50),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
        cv2.imshow("Outpput",theimage)
    else:
        theimage = cv2.putText(cv2.resize(img, (256, 256)), "Good Image", (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
        cv2.imshow("Outpput", theimage)

# This function classifies  multiple images, it takes in the path to a collection of images  and the saved model
def classifymoreimages(imagespath, model):
    model = load_model(model)
    theimage = np.ones((1000,1000,3),dtype=np.uint8)

    path = imagespath

    theimages = next(os.walk(path))[2]
    y = 30
    writeup = ""
    for i in theimages:
        imagepath = os.path.join(path, i)
        img = cv2.imread(imagepath)
        resizedImage = cv2.resize(img, (128, 128))
        imagenp = np.array(resizedImage / 255)
        expanded = np.expand_dims(imagenp, axis=0)
        output = model.predict(expanded)
        if (output[0][0] < 0.5):
            writeup =  i + "- Bad Image"
            cv2.putText(theimage, writeup, (20, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 1)
        if (output[0][0] > 0.5):
            writeup = i + "- Good Image"
            cv2.putText(theimage, writeup, (20, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 1)
        y = 25 + y
    cv2.imshow("Outpput", theimage)




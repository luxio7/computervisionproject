import os
from skimage import io
from skimage.transform import resize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from keras import optimizers
from keras.models import Sequential 
import pickle
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
import sys
import random
import datetime
from lxml import etree

# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "D:/computervisiondatabase/VOCtrainval_11-May-2009/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 144    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)
                    # neem iets dat deelbaar is door 8
if False:
    # step1 - build list of filtered filenames
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []
    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
            filtered_filenames.append(a_f[:-4])
    
    # step2 - build (x,y) for TRAIN/VAL (classification)
    classes_folder = os.path.join(voc_root_folder, "VOC2009/ImageSets/Main/")
    classes_files = os.listdir(classes_folder)
    train_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_train.txt' in c_f]
    val_files = [os.path.join(classes_folder, c_f) for filt in filter for c_f in classes_files if filt in c_f and '_val.txt' in c_f]
    
    
    def build_classification_dataset(list_of_files):
        """ build training or validation set
    
        :param list_of_files: list of filenames to build trainset with
        :return: tuple with x np.ndarray of shape (n_images, image_size, image_size, 3) and  y np.ndarray of shape (n_images, n_classes)
        """
        temp = []
        train_labels = []
        for f_cf in list_of_files:
            with open(f_cf) as file:
                lines = file.read().splitlines()
                temp.append([line.split()[0] for line in lines if int(line.split()[-1]) == 1])
                label_id = [f_ind for f_ind, filt in enumerate(filter) if filt in f_cf][0]
                train_labels.append(len(temp[-1]) * [label_id])
        train_filter = [item for l in temp for item in l]
    
        image_folder = os.path.join(voc_root_folder, "VOC2009/JPEGImages/")
        image_filenames = [os.path.join(image_folder, file) for f in train_filter for file in os.listdir(image_folder) if
                           f in file]
        x = np.array([resize(io.imread(img_f), (image_size, image_size, 3)) for img_f in image_filenames]).astype(
            'float32')
        # changed y to an array of shape (num_examples, num_classes) with 0 if class is not present and 1 if class is present
        y_temp = []
        for tf in train_filter:
            y_temp.append([1 if tf in l else 0 for l in temp])
        y = np.array(y_temp)
    
        return x, y
    
    
    x_train, y_train = build_classification_dataset(train_files)
    print('%i training images from %i classes' %(x_train.shape[0], y_train.shape[1]))
    x_val, y_val = build_classification_dataset(val_files)
    print('%i validation images from %i classes' %(x_val.shape[0],  y_train.shape[1]))
    
    # ------------------------
    
    # store output -> dat elke keer maken duurt te lang
    
# =============================================================================
#     io.imshow(x_val[0])
#     plt.show()
#     sys.exit()
# =============================================================================
    
    # Saving the objects:
    with open('x_train.txt', 'wb') as f:
        pickle.dump(x_train, f)
    with open('y_train.txt', 'wb') as f:
        pickle.dump(y_train, f)
    with open('x_val.txt', 'wb') as f:
        pickle.dump(x_val, f)
    with open('y_val.txt', 'wb') as f:
        pickle.dump(y_val, f)
        
else:
    # Getting back the objects:
    with open('x_train.txt', 'rb') as f:
        x_train = pickle.load(f)
    with open('y_train.txt', 'rb') as f:
        y_train = pickle.load(f)
    with open('x_val.txt', 'rb') as f:
        x_val = pickle.load(f)
    with open('y_val.txt', 'rb') as f:
        y_val = pickle.load(f)

# from here, you can start building your model
# you will only need x_train and x_val for the autoencoder
# you should extend the above script for the segmentation task (you will need a slightly different function for building the label images)

#handige links
# https://www.youtube.com/watch?v=hbU7nbVDzGE PCA vs auto-encoders
# https://blog.keras.io/building-autoencoders-in-keras.html voorbeeldcode
# https://stackoverflow.com/questions/48243360/how-to-determine-the-filter-parameter-in-the-keras-conv2d-function filtersize
# https://stackoverflow.com/questions/51877834/fine-tuning-of-keras-autoencoders-of-cat-images
# https://github.com/shibuiwilliam/Keras_Autoencoder/blob/master/Cifar_Conv_AutoEncoder.ipynb
# https://towardsdatascience.com/a-guide-to-an-efficient-way-to-build-neural-network-architectures-part-ii-hyper-parameter-42efca01e5d7

autoencoder = None
'''
input_img = Input(shape=(image_size, image_size, 3))  # adapt this if using `channels_first` image data format, origineel (28,28,1) dit wilt zeggen 28 breed 28 hoog en 1 diep (zwart wit dus)

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) #eerste argument is the dimensionality of the output space (i.e. the number of output filters in the convolution).
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (27, 27, 8) i.e. 128-dimensional

# grotere filter (5 op 5 bijvoorbeeld)
# groter aantal filter

x = Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x) #tegenovergestelde van MaxPooling2D
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy']) #binary_crossentropy kan ook maar traint langer per keer



#netwerk van 0 trainen
if False:
    history = autoencoder.fit(x_train, x_train,
                    epochs=1000,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_val, x_val))
    tijd = datetime.datetime.now()
    tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
    autoencoder.save_weights("data/weights"+tijdstring+".h5")
    
    print(history.history.keys())
    # summarize history for accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
else:
    #netwerk nog verder trainen
    if False:
        autoencoder.load_weights("data/weights.h5")
        autoencoder.fit(x_train, x_train,
                    epochs=100,
                    batch_size=64,
                    shuffle=True,
                    validation_data=(x_val, x_val))
        tijd = datetime.datetime.now()
        tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
        autoencoder.save_weights("data/weights"+tijdstring+".h5")
    else:
        autoencoder.load_weights("data/weights.h5")




#vergelijk input met output
io.imshow(x_val[0])
plt.show()
testPicture = np.asarray([x_val[0].tolist()])
uitkomst = autoencoder.predict(testPicture)[0]
io.imshow(uitkomst)
plt.show()
'''

# =============================================================================
# CLASSIFICATIE (deel 3)
# logic regression
# https://towardsdatascience.com/understanding-logistic-regression-9b02c2aec102
#  -> 'As discussed earlier, to deal with outliers, Logistic Regression uses Sigmoid function.'
#
# fine-tuning
# https://www.learnopencv.com/keras-tutorial-fine-tuning-using-pre-trained-models/
#
# loss function
# https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
#  -> model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
# ik denk dit -> model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# =============================================================================

#-------------------------
#EERSTE MODEL
# https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/
# https://stackoverflow.com/questions/42081257/keras-binary-crossentropy-vs-categorical-crossentropy-performance
# https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/
#-------------------------
if False:
    # https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
    #autoencoder.load_weights("data/weights.h5")
    #encoderlayers = autoencoder.layers[:8] # geen idee of dit werkt
    
    '''
    # Freeze the layers
    for i in range(1,8,2):
        encoderlayers[i].trainable = False
    '''
    
    input_dim = 10368 #18*18*64
    output_dim = nb_classes = 5 
    
    for i in range(len(autoencoder.layers)):
        autoencoder.layers[i].trainable = False
    x = autoencoder.layers[12].output

    x = Flatten()(x)
    x = Dense(2048, input_dim = input_dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, input_dim = 4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output1 = Dense(1, input_dim = 1024, activation = 'sigmoid')(x)
    output2 = Dense(1, input_dim = 1024, activation = 'sigmoid')(x)
    output3 = Dense(1, input_dim = 1024, activation = 'sigmoid')(x)
    output4 = Dense(1, input_dim = 1024, activation = 'sigmoid')(x)
    output5 = Dense(1, input_dim = 1024, activation = 'sigmoid')(x)

    modelPredict = Model(autoencoder.inputs,[output1,output2,output3,output4,output5])    
    
    plot_model(modelPredict, to_file='model_plot_modelPredict.png', show_shapes=True, show_layer_names=True)
    modelPredict.summary()
    
    modelPredict.compile(optimizer = "adam", loss = ["mse","mse","mse","mse","mse"], metrics=[])
    #train van 0
    if True:
        y_train_parts = []
        y_val_parts = []
        for i in range (0,5):
            y_train_parts.append(y_train[:,i].ravel())
            y_val_parts.append(y_val[:,i].ravel())
        
        history = modelPredict.fit(x_train, y_train_parts,shuffle=True, batch_size=64, nb_epoch=50,verbose=1, validation_data=(x_val, y_val_parts))
        tijd = datetime.datetime.now()
        tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
        modelPredict.save_weights("data/weightsdeel31new"+tijdstring+".h5")
        
        print(history.history.keys())
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    else:
        modelPredict.load_weights("data/weightsdeel31.h5")
    
    predictedValues = modelPredict.predict(x_val)
    
    predictedValues = np.asarray(predictedValues)
    predictedValues = np.transpose(predictedValues)
    predictedValuesBU = np.copy(predictedValues)
    
    # accuracy test
    hoogsteClassificatie = np.apply_along_axis(np.argmax,1,predictedValues[0])
    
    juistVoorspeld = 0
    for i,row in enumerate(y_val):
        if row[hoogsteClassificatie[i]] == 1:
            juistVoorspeld += 1
    accuracy = juistVoorspeld/len(y_val)
    print("accuracy is: " + str(accuracy))
    
    for _ in range (0,50):
        i = random.randint(0, len(y_val))
        io.imshow(x_val[i])
    
        text = ""
        for j,classe in enumerate(filter):
            text = text + classe + ": " + str(round(predictedValuesBU[0,i,j],2)) + "\n"
        for k,l in enumerate(y_val[i]):
            if l == 1:
                text = text + " " + filter[k] + "\n"
            
        plt.xlabel(text)
        plt.show()
    
#-------------------------
#TWEEDE MODEL
#-------------------------
if True:
    # https://medium.com/@vijayabhaskar96/multi-label-image-classification-tutorial-with-keras-imagedatagenerator-cd541f8eaf24
    #encoderlayers = autoencoder.layers[:8] # geen idee of dit werkt
    
    '''
    # Freeze the layers
    for i in range(1,8,2):
        encoderlayers[i].trainable = False
    '''
    
    input_dim = 10368 #8*8*64
    output_dim = nb_classes = 5 
    
    input_img = Input(shape=(image_size, image_size, 3))  # adapt this if using `channels_first` image data format, origineel (28,28,1) dit wilt zeggen 28 breed 28 hoog en 1 diep (zwart wit dus)
    
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(input_img) #eerste argument is the dimensionality of the output space (i.e. the number of output filters in the convolution).
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='random_uniform')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    x = Flatten()(encoded)
    x = Dense(2048, input_dim = input_dim, activation='relu', kernel_initializer='random_uniform')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, input_dim = 2048, activation='relu', kernel_initializer='random_uniform')(x)
    x = Dropout(0.5)(x)
    
    output1 = Dense(1, input_dim = 1024, activation = 'sigmoid', kernel_initializer='random_uniform')(x)
    output2 = Dense(1, input_dim = 1024, activation = 'sigmoid', kernel_initializer='random_uniform')(x)
    output3 = Dense(1, input_dim = 1024, activation = 'sigmoid', kernel_initializer='random_uniform')(x)
    output4 = Dense(1, input_dim = 1024, activation = 'sigmoid', kernel_initializer='random_uniform')(x)
    output5 = Dense(1, input_dim = 1024, activation = 'sigmoid', kernel_initializer='random_uniform')(x)

    modelPredict = Model(input_img,[output1,output2,output3,output4,output5])    
    modelPredict.summary()
    modelPredict.compile(optimizer = "adam", loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy"], metrics=[])
    
    y_train_parts = []
    y_val_parts = []
    for i in range (0,5):
        y_train_parts.append(y_train[:,i].ravel())
        y_val_parts.append(y_val[:,i].ravel())
    
    history = modelPredict.fit(x_train, y_train_parts, batch_size=64,shuffle=True, nb_epoch=200,verbose=1, validation_data=(x_val, y_val_parts))
    
    print(history.history.keys())
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    
    tijd = datetime.datetime.now()
    tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
    modelPredict.save_weights("data/weightsdeel31new"+tijdstring+".h5")
    
    predictedValues = modelPredict.predict(x_val)
    
    predictedValues = np.asarray(predictedValues)
    predictedValues = np.transpose(predictedValues)
    predictedValuesBU = np.copy(predictedValues)
    
    # accuracy test
    hoogsteClassificatie = np.apply_along_axis(np.argmax,1,predictedValues[0])
    
    juistVoorspeld = 0
    for i,row in enumerate(y_val):
        if row[hoogsteClassificatie[i]] == 1:
            juistVoorspeld += 1
    accuracy = juistVoorspeld/len(y_val)
    print("accuracy is: " + str(accuracy))
    
    for _ in range (0,50):
        i = random.randint(0, len(y_val))
        io.imshow(x_val[i])
    
        text = ""
        for j,classe in enumerate(filter):
            text = text + classe + ": " + str(round(predictedValuesBU[0,i,j],2)) + "\n"
        for k,l in enumerate(y_val[i]):
            if l == 1:
                text = text + " " + filter[k] + "\n"
            
        plt.xlabel(text)
        plt.show()



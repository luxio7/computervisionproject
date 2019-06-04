import os
from skimage import io
from skimage.transform import resize
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Activation, Dropout
from keras.models import Model
from keras.utils import to_categorical
from keras import backend as K
import numpy as np
from keras import optimizers
import pickle
from matplotlib import pyplot as plt
from keras.utils.vis_utils import plot_model
import random
import sys
import datetime
from lxml import etree

# parameters that you should set before running this script
filter = ['aeroplane', 'car', 'chair', 'dog', 'bird']       # select class, this default should yield 1489 training and 1470 validation images
voc_root_folder = "D:/computervisiondatabase/VOCtrainval_11-May-2009/VOCdevkit/"  # please replace with the location on your laptop where you unpacked the tarball
image_size = 144    # image size that you will use for your network (input images will be resampled to this size), lower if you have troubles on your laptop (hint: use io.imshow to inspect the quality of the resampled images before feeding it into your network!)
                    # image_size should be divisible by 8 for the best result
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
    
# ^^^ Given code ^^^
# -----------------------------------------------------------------------------
# ⌄⌄⌄ Own code ⌄⌄⌄
    
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


# =============================================================================
# 1. Constructing the auto-encoder
#    - This auto-encoder model will take (image_size, image_size, 3) as input
#      and will output an image of the same size.
#    - The encoded layer has size (image_size/8,image_size/8,32) and will thus
#      reduce the imput by a factor 6.
# =============================================================================
input_img = Input(shape=(image_size, image_size, 3))

x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) 
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
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


# =============================================================================
# 2. Training the model
#    - When trainModel is TRUE, the model will learn how to encode pictures 
#      by encoding and decoding the train images and try to learn a way that 
#      minimizes the mean square error.
#    - The weights are stored in the data folder.
#    - The training metrics accuracy, loss, validation accuracy and validation
#      accuracy are plotted when the training is done.
#    - When loadModelFromDisk is TRUE, the model will load previously trained 
#      weights and possibly train them further based on 
#      the trainModel parameter.
#    - When the weights are loaded in a picture is encoded and decoded to see
#      the result.
# =============================================================================
trainModel = False
loadModelFromDisk = True

if loadModelFromDisk:
    autoencoder.load_weights("data/weights.h5")
if trainModel:
    history = autoencoder.fit(x_train, x_train,
                    epochs=2,
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
if ((not trainModel) and (not loadModelFromDisk)):
    print('model is untrained, part runPart3_1 will be skipped')
    runPart3_1 = False
else:
    # show result of the model
    io.imshow(x_val[0])
    plt.show()
    testPicture = np.asarray([x_val[0].tolist()])
    uitkomst = autoencoder.predict(testPicture)[0]
    io.imshow(uitkomst)
    plt.show()

# =============================================================================
# CLASSIFICATION
# - Part3_1 is a model that takes the encoder from the previous auto-encoder 
#   and sticks 2 fully connected layers that will preform the task of classifier
#       - When runPart3_1 is True the model will be constructed and when
#         trainPart3_1 is True it will train the model from scratch, when it is
#         False, the weights are loaded in. NOTE: this model can only train
#         weights of the new layers. The encoder layers still have their 
#         weights from the previous section.
# - Part3_2 is the same model as 3_1 but now all the weights are trainable
# - In both cases, training metrics will be printed, newly traind weights are
#   Stored to disk and the model is tested on 20 randomly selected validation
#   images. Those results will be printed in the console.
# =============================================================================
runPart3_1 = True
trainPart3_1 = False
runPart3_2 = True
trainPart3_2 = False

input_dim = 10368 #18*18*32
output_dim = nb_classes = len(filter)

y_train_parts = []
y_val_parts = []
for i in range (0,nb_classes):
    helper = y_train[:,i].ravel()
    helper2 = helper.copy()
    helper = np.append(helper,helper2,axis=0)
    y_train_parts.append(helper)
    y_val_parts.append(y_val[:,i].ravel())

x_trainMirror = x_train[...,::-1,:].copy()
x_train = np.append(x_train,x_trainMirror, axis=0)

if runPart3_1:
    
    for i in range(len(autoencoder.layers)):
        autoencoder.layers[i].trainable = False
        
    x = autoencoder.layers[12].output

    x = Flatten()(x)
    x = Dense(2048, input_dim = input_dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, input_dim = 2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output1 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output2 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output3 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output4 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output5 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)

    modelPredict = Model(autoencoder.inputs,[output1,output2,output3,output4,output5])    
    
    plot_model(modelPredict, to_file='model_plot_modelPredict.png', show_shapes=True, show_layer_names=True)
    modelPredict.summary()
    
    modelPredict.compile(optimizer = "adam", loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy"], metrics=[])
    
    
    if trainPart3_1:
        history = modelPredict.fit(x_train, y_train_parts,shuffle=True, batch_size=64, nb_epoch=30,verbose=1, validation_data=(x_val, y_val_parts))
        tijd = datetime.datetime.now()
        tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
        modelPredict.save_weights("data/weightsdeel31"+tijdstring+".h5")
        
        print(history.history.keys())
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
    highestClassification = np.apply_along_axis(np.argmax,1,predictedValues[0])
    
    predictedRight = 0
    for i,row in enumerate(y_val):
        if row[highestClassification[i]] == 1:
            predictedRight += 1
    accuracy = predictedRight/len(y_val)
    print("accuracy is: " + str(accuracy))
    
    for _ in range (0,20):
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
#SECOND MODEL
#-------------------------
if runPart3_2:
    
    input_img = Input(shape=(image_size, image_size, 3))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img) 
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
    
    x = Flatten()(encoded)
    x = Dense(2048, input_dim = input_dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(2048, input_dim = 2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    output1 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output2 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output3 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output4 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)
    output5 = Dense(1, input_dim = 2048, activation = 'sigmoid')(x)

    modelPredict = Model(input_img,[output1,output2,output3,output4,output5])    
    modelPredict.summary()
    modelPredict.compile(optimizer = "adam", loss = ["binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy","binary_crossentropy"], metrics=[])


    if trainPart3_2:
        history = modelPredict.fit(x_train, y_train_parts, batch_size=64,shuffle=True, nb_epoch=6,verbose=1, validation_data=(x_val, y_val_parts))
        
        print(history.history.keys())
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
        tijd = datetime.datetime.now()
        tijdstring = tijd.strftime("%H:%M:%S").replace(":", "_")
        modelPredict.save_weights("data/weightsdeel32"+tijdstring+".h5")
    else:
        modelPredict.load_weights("data/weightsdeel32.h5")
    
    predictedValues = modelPredict.predict(x_val)
    predictedValues = np.asarray(predictedValues)
    predictedValues = np.transpose(predictedValues)
    predictedValuesBU = np.copy(predictedValues)
    
    # accuracy test
    highestClassification = np.apply_along_axis(np.argmax,1,predictedValues[0])
    
    predictedRight = 0
    for i,row in enumerate(y_val):
        if row[highestClassification[i]] == 1:
            predictedRight += 1
    accuracy = predictedRight/len(y_val)
    print("accuracy is: " + str(accuracy))
    
    for _ in range (0,20):
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



# -*- coding: utf-8 -*-
"""
Created on Sat Aug 12 2023

@author: Achrafkr
"""

# Imports 

import numpy as np
import os
import tensorflow as tf

from image_tools import generate_dataset, baseline_model, plot_history, model_accuracy, plot_confusion_matrix

def classify(train_repertory, test_repertory, train_params, pretrained_model = True, load_weights = False):
    
    # Predefined variables
    
    class_names = os.listdir(path_train)
    image_size = (64, 64)
    batch_size = 64
    validation_split = 0.17
    
    # Load data
    print("Loading data")
    
    train_data = generate_dataset(train_repertory, image_size, batch_size, validation_split, 'training', class_names)
    val_data = generate_dataset(train_repertory, image_size, batch_size,validation_split, 'validation', class_names)
    
    test_data = generate_dataset(test_repertory, image_size, batch_size, None, None, class_names)
    
    # Prefetching data for boosting performance
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_data = train_data.prefetch(AUTOTUNE)
    val_data = val_data.prefetch(AUTOTUNE)
    test_data = test_data.prefetch(AUTOTUNE)
    
    print("Data Processing finished")
    
    if pretrained_model:
        model_directory = train_params['model_params']['model_directory']
        model_name = train_params['model_params']['model_name']
        model = tf.keras.models.load_model(model_directory + model_name)
        
        if load_weights:
            weights_directory = train_params['weights']['weights_directory_l']
            model.load_weights(weights_directory)
        
    else:
        
        model = train_params['model_params']['model']
    
        if load_weights:
            weights_directory = train_params['weights']['weights_directory_l']
            model.load_weights(weights_directory)
        
        epochs = train_params['epochs']
        
        optimizer = train_params['optimizer']
        loss = train_params['loss']
            
            
        if train_params['callbacks']['save_callbacks']:
            
            callbacks_directory = train_params['callbacks']['callbacks_directory']
            hist_filename = train_params['callbacks']['hist_filename']
            checkpoint_filename = train_params['callbacks']['checkpoint_filename']
            
            callbacks = [tf.keras.callbacks.CSVLogger(callbacks_directory + hist_filename, separator=",", append=True),
                     tf.keras.callbacks.ModelCheckpoint(callbacks_directory + checkpoint_filename + "{epoch}.keras")]
            
        else:
            callbacks = [None]
        
        model.compile(optimizer=optimizer,
            loss=loss,
            metrics=["accuracy"])
        
        print("Start training")
        
        history = model.fit(train_data,
                    epochs=epochs,
                    callbacks = callbacks,
                    validation_data=val_data)
        
        print("Training finished")

        if train_params['model_params']['save_model']:
            model_name = train_params['model_params']['model_name']
            model_directory = train_params['model_params']['model_directory']
            model.save(model_directory + model_name)
            
            print("Model parameters saved")
            
        if train_params['weights']['save_weights']:
            weights_directory = train_params['weights']['weights_directory_s']
            weights_filename = train_params['weights']['weights_filename']
            model.save_weights(weights_directory + weights_filename)
            
            print("Model weights saved")
            
    input_, target_ = np.array(list(map(lambda x: [x[0], x[1].numpy()], test_data.unbatch())), dtype = object).T
    predictions = np.array([np.argmax(model.predict(np.expand_dims(input_[i], 0))) for i in range(len(input_))])

    return model, input_, target_, predictions


if __name__ == "__main__":
    
    path_train = "../Data/EuroSat/2750/Training/"
    path_test = "../Data/EuroSat/2750/Test/"
    
    class_names = os.listdir(path_train)
    image_size = (64, 64)
    
    pretrained_model = True
    
    if pretrained_model:
        
        model_params = {'model_name': 'baseline_model', 
                        'model_directory':"../Models/", 
                        'save_model': False
                        }
        
        train_params = {'model_params': model_params, 'weights': None}
        
    
    # Define model and training parameters
    if not pretrained_model:
    
        CNN_model = baseline_model(input_shape = image_size + (3,), num_classes = len(class_names))
        
        model_params = {'model': CNN_model, 
                        'model_name': 'CNN_model', 
                        'model_directory':"../Models/test/", 
                        'save_model': True
                        }
        
        weights = {'load_weights': True, 
                   'weights_directory_l': "../Models/test/baseline_model_30_epochs", 
                   'save_weights': True, 
                   'weights_directory_s': "../Models/test/", 
                   'weights_filename': 'new_weights_epoch_1'
                   }
    
        callbacks = {'save_callbacks': True,
                     'callbacks_directory': "../Models/test/",
                     'hist_filename': "test_history.csv",
                     'checkpoint_filename': "checkpoint_epoch_"
                     }
        
        epochs = 1
        optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-3)
        loss = 'SparseCategoricalCrossentropy'
    
        train_params = {'model_params': model_params, 'weights': weights, 'callbacks': callbacks, 'epochs': epochs,
                        'optimizer': optimizer, 'loss': loss}


    # Train model and make predictions
    #model, X, y, y_pred = classify(path_train, path_test, train_params, pretrained_model, load_weights = True)
    model, X, y, y_pred = classify(path_train, path_test, train_params, pretrained_model, load_weights = False)
    
    acc, _ = model_accuracy(model, [X, y, y_pred])


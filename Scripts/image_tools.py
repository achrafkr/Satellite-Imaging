# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 2023

@author: Achrafkr
"""

""" 
Imports
"""
    
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Classification
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Segmentation
from sklearn.cluster import KMeans
import cv2
from shapely.geometry.polygon import Polygon
from shapely.ops import unary_union 


""" 
Data and Image processing
"""
def generate_dataset(directory, image_size, batch_size, validation_split, subset, class_names, labels = 'inferred', \
                    label_mode = 'int', shuffle = True):
    
    data = image_dataset_from_directory(directory,
    labels=labels,
    label_mode=label_mode,
    class_names=class_names,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size,
    shuffle=shuffle,
    seed=0,
    validation_split=validation_split,
    subset=subset,
    interpolation='bilinear',
    follow_links=False)
    
    return data

def draw_contour(image, thresh, t_method = 'binary'):
    image_size = image.shape
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    method = {'binary': cv2.THRESH_BINARY, 'binary_inv': cv2.THRESH_BINARY_INV, 
              'trunc': cv2.THRESH_TRUNC, 'trozero': cv2.THRESH_TOZERO, 'trozero_inv': cv2.THRESH_TOZERO_INV}
    
    _, thresh_img = cv2.threshold(gray_img, thresh, 255, method[t_method])
    contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_contours = np.zeros(image_size)
    contour_img = cv2.drawContours(img_contours, contours, -1, (102, 204, 0), 1)
    
    return contour_img, contours

def polygon_from_contour(contour):
    List_polygons = []
    for cont in contour:
        try:
            List_polygons.append(Polygon(cont.reshape(-1, 2)))
        except ValueError:
            #print(f"{cont.reshape(-1, 2)} is not a polygon")
            print("Can't form a polygon from 2D line")
            
    return unary_union(List_polygons)

def compute_area(contours, return_pct = False, image_size = None):
    contour = polygon_from_contour(contours)
    
    if return_pct:
        x, y, _ = image_size if len(image_size) == 3 else image_size + (None,)
        frame = Polygon([(0, 0), (0, y), (x, y), (x, 0)])
        
        return np.round(100 * contour.area / frame.area, 4)
    
    return np.round(contour.area, 4)

""" 
Models
"""
# Classification

def baseline_model(input_shape, num_classes):
    inputs = tf.keras.Input(shape=input_shape)

    # Entry block
    x = tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs) # Rescaling
    x = tf.keras.layers.Conv2D(128, 3, strides=2, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [256, 512, 728]:
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.SeparableConv2D(size, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        x = tf.keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = tf.keras.layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = tf.keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    x = tf.keras.layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(units, activation=activation)(x)
    return tf.keras.Model(inputs, outputs)

# Clustering

def clustering_km(image, n_clusters):
    img = image.reshape((-1, 1))
    cluster = KMeans(n_clusters = n_clusters, n_init = 4)
    cluster.fit(img)
    centroids = cluster.cluster_centers_.squeeze()
    labels = cluster.labels_
    
    return centroids, labels

def cv2_clustering_km(image, n_clusters, eps, max_iter = 10, attempts = 10):
    image_size = image.shape
    img = image.reshape((-1, 3)).astype('float32')
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    _, labels, centroids = cv2.kmeans(img, n_clusters, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS) 
    centroids = np.uint8(centroids)
    segmented_data = centroids[labels.flatten()]
    segmented_image = segmented_data.reshape(image_size)
    
    return segmented_image

""" 
Metrics
"""
def model_accuracy(model, data):
    
    if str(type(data)) == "<class 'tensorflow.python.data.ops.dataset_ops._UnbatchDataset'>":
        input_, target_ = np.array(list(map(lambda x: [x[0], x[1].numpy()], data)), dtype = object).T
        predictions = np.array([np.argmax(model.predict(np.expand_dims(input_[i], 0))) for i in range(len(input_))])
        
    elif type(data) == list:
        
        if len(data) == 3:
            input_, target_, predictions = data
            
        elif len(data) == 2:
            input_, target_ = data
            predictions = np.array([np.argmax(model.predict(np.expand_dims(input_[i], 0))) for i in range(len(input_))])
            
        else:
            raise ValueError("Only dimensions 2 and 3 are allowed for data object")
        
    else:
        raise ValueError("Unknown object: use list or 'tensorflow.python.data.ops.dataset_ops._UnbatchDataset' object instead")
    
    accuracy = np.sum(target_ == predictions)/len(target_)
    print("Accuracy: {:.3f}".format(accuracy))
    n_classes = len(set(target_))
    c_matrix = np.zeros((n_classes, n_classes))
    for t, p in zip(target_, predictions):
        c_matrix[t, p] += 1
        
    return accuracy, c_matrix

""" 
Display
"""
def plot_history(history):
    _, ax = plt.subplots(1, 2, figsize = (14, 3.7))

    plt.suptitle('Loss and accuracy evolution')
    ax[0].plot(history.history['loss'], color = 'slateblue', label = 'Training loss')
    ax[0].plot(history.history['val_loss'], color = 'orangered', label = 'Validation loss', alpha = .7)
    ax[0].legend()
    ax[0].set_ylabel('Loss')
    ax[0].set_xlabel('Epoch')

    ax[1].plot(history.history['accuracy'], color = 'slateblue', label = 'Training accuracy')
    ax[1].plot(history.history['val_accuracy'], color = 'orangered', label = 'Validation accuracy', alpha = .7)
    ax[1].legend()
    ax[1].set_ylabel('Accuracy')
    ax[1].set_xlabel('Epoch')

    plt.show()
    
def plot_confusion_matrix(c_matrix, labels, cmap = 'GnBu', figsize = (6, 6), cbar = False, fmt = '.2g', ax_ = None):
    if ax_ != None:
        ax = ax_
    else:
        _, ax = plt.subplots(figsize = (figsize))
        
    sns.heatmap(c_matrix, cmap = cmap, annot = True, cbar = cbar, fmt = fmt, ax = ax)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_xticks(np.arange(0.5, len(labels) + 0.5), labels, rotation = 90)
    ax.set_yticks(np.arange(0.5, len(labels) + 0.5), labels, rotation = 0)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    
    plt.show()
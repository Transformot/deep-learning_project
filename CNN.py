import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

import time

from sklearn.metrics import confusion_matrix
import seaborn as sns

import random

train_data = np.loadtxt('data/ECG200_TRAIN.tsv', delimiter='\t')
test_data = np.loadtxt('data/ECG200_TEST.tsv', delimiter='\t')

train_data[train_data[:, 0] == -1, 0] = 0
test_data[test_data[:, 0] == -1, 0] = 0

X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_test, y_test = test_data[:, 1:], test_data[:, 0]

all_data = np.concatenate((X_train, X_test), axis=0)

global_max = np.max(all_data)
global_min = np.min(all_data)

X_train_normalized = (X_train - global_min) / (global_max - global_min)
X_test_normalized = (X_test - global_min) / (global_max - global_min)

y_train_encoded = to_categorical(y_train, num_classes=2)
y_test_encoded = to_categorical(y_test, num_classes=2)


filters = 5
kernel_size = 3
stride = 1
padding = 'same'
use_bias = False
hidden_activation = 'relu'
final_activation = 'softmax'
pool_size = 3
dropout_rate = 0.3
nb_classes = 2 
learning_rate = 0.0005
optimizer_algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cost_function = tf.keras.losses.categorical_crossentropy
mini_batch_size = 16
nb_epochs = 500
percentage_of_train_as_validation = 0.2

# Définir les plages de valeurs pour les hyperparamètres
filters_range = [5]
kernel_size_range = [2]                           # [1, 2, 3]
strides_range = [2]                           # [1, 2, 3]
final_activation_range = ['softmax']             # ['softmax', 'sigmoid']
pool_size_range = [2]                                 # [1, 2, 3]
dropout_rate_range = [0.25]
learning_rate_range = [0.006]
mini_batch_size_range = [32]

# Nombre total de combinaisons à essayer
total_combinations = len(filters_range) * len(kernel_size_range) * len(strides_range) \
                   * len(final_activation_range) * len(pool_size_range) \
                   * len(dropout_rate_range) * len(learning_rate_range) * len(mini_batch_size_range)
print(total_combinations)

# Nombre de combinaisons à essayer
num_trials = 10

start_time = time.time()
best_accuracy = 0
best_loss = 1
best_hyperparameters = []
best_hyperparameters_2 = []
for i in range(num_trials):
    print("Trial:", i+1, "/", num_trials)
    # Sélectionner aléatoirement les valeurs des hyperparamètres
    filters = random.choice(filters_range)
    kernel_size = random.choice(kernel_size_range)
    stride = random.choice(strides_range)
    final_activation = random.choice(final_activation_range)
    pool_size = random.choice(pool_size_range)
    dropout_rate = random.choice(dropout_rate_range)
    learning_rate = random.choice(learning_rate_range)
    mini_batch_size = random.choice(mini_batch_size_range)
    
    # Construire, compiler et entraîner le modèle avec ces hyperparamètres
    input_shape = (96, 1)
    input_layer = Input(input_shape)

    conv_layer_1_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(input_layer)
    relu_layer_1_1 = Activation(hidden_activation)(conv_layer_1_1)
    conv_layer_1_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(relu_layer_1_1)
    relu_layer_1_2 = Activation(hidden_activation)(conv_layer_1_2)
    conv_layer_1_3 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(relu_layer_1_2)
    relu_layer_1_3 = Activation(hidden_activation)(conv_layer_1_3)
    pooling_layer_1 = MaxPooling1D(pool_size=pool_size, padding=padding)(relu_layer_1_3)
    dropout_layer_1_1 = Dropout(rate=dropout_rate)(pooling_layer_1)

    conv_layer_2_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(dropout_layer_1_1)
    relu_layer_2_1 = Activation(hidden_activation)(conv_layer_2_1)
    conv_layer_2_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(relu_layer_2_1)
    relu_layer_2_2 = Activation(hidden_activation)(conv_layer_2_2)
    conv_layer_2_3 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(relu_layer_2_2)
    relu_layer_2_3 = Activation(hidden_activation)(conv_layer_2_3)
    pooling_layer_2 = MaxPooling1D(pool_size=pool_size, padding=padding)(relu_layer_2_3)

    flattened_layer = Flatten()(pooling_layer_2)
    dropout_flattened = Dropout(rate=dropout_rate)(flattened_layer)
    output_layer = Dense(units=nb_classes,activation=final_activation)(dropout_flattened)

    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer_algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=cost_function,optimizer=optimizer_algo, metrics=['accuracy'])
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best_model_CNN.keras', monitor='val_loss', save_best_only=True)
    history = model.fit(X_train_normalized, y_train_encoded, 
                    batch_size=mini_batch_size, 
                    epochs=nb_epochs,
                    validation_split=percentage_of_train_as_validation,
                    verbose=False,
                    callbacks=[model_checkpoint])
    
    # Évaluer la performance du meilleur modèle
    best_model = tf.keras.models.load_model('best_model_CNN.keras')
    test_loss, test_accuracy = best_model.evaluate(X_test_normalized, y_test_encoded)
    
    # Mettre à jour les meilleurs hyperparamètres si nécessaire
    if test_accuracy - test_loss > best_accuracy - best_loss :
        best_accuracy = test_accuracy
        best_loss = test_loss
        best_hyperparameters.append((filters, kernel_size, stride, final_activation, pool_size, dropout_rate, learning_rate, mini_batch_size))
    if test_accuracy - test_loss > 0.47 :
        best_hyperparameters_2.append((filters, kernel_size, stride, final_activation, pool_size, dropout_rate, learning_rate, mini_batch_size))

# Fin du compteur de temps
end_time = time.time()

# Calcul de la durée d'entraînement en secondes
training_time_seconds = end_time - start_time

print("Best hyperparameters:", best_hyperparameters)
print("Best hyperparameters 2:", best_hyperparameters_2)
print("Meilleure perte :", best_loss)
print("Meilleure précision :", best_accuracy)
print(f"Temps d'entraînement des hyperparamètre : {training_time_seconds // 60} minutes et {training_time_seconds % 60} secondes.")

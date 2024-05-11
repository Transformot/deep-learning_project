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
activation = 'relu'
pool_size = 3
dropout_rate = 0.3
nb_classes = 2 
learning_rate = 0.0005
optimizer_algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cost_function = tf.keras.losses.categorical_crossentropy
mini_batch_size = 16
nb_epochs = 800
percentage_of_train_as_validation = 0.2

# Définir les plages de valeurs pour les hyperparamètres
filters_values = [3, 5, 10, 15]
kernel_size_values = [1, 2, 3, 5, 7]
strides_values = [1, 2, 3, 5]
pool_size_values = [1, 2, 3, 5]
dropout_rate_values = [0.1, 0.2, 0.3, 0.5, 0.7]
learning_rate_values = [0.001, 0.0001, 0.00001]
mini_batch_size_values = [8, 16, 32, 64]

# Nombre total de combinaisons à essayer
total_combinations = len(filters_values) * len(kernel_size_values) * len(strides_values) \
                   * len(pool_size_values) * len(dropout_rate_values) * len(learning_rate_values) \
                   * len(mini_batch_size_values)
print(total_combinations)
# Nombre de combinaisons à essayer
num_trials = 60

# Début du compteur de temps
start_time = time.time()

# Effectuer une recherche aléatoire
best_accuracy = 0
best_hyperparameters = None
for i in range(num_trials):
    print("epoch ", i)
    print("on ", num_trials)
    # Sélectionner aléatoirement les valeurs des hyperparamètres
    filters = random.choice(filters_values)
    kernel_size = random.choice(kernel_size_values)
    stride = random.choice(strides_values)
    pool_size = random.choice(pool_size_values)
    dropout_rate =random.choice(dropout_rate_values)
    learning_rate = random.choice(learning_rate_values)
    mini_batch_size =random.choice(mini_batch_size_values)
    
    # Construire, compiler et entraîner le modèle avec ces hyperparamètres
    input_shape = (96, 1)
    input_layer = Input(input_shape)
    conv_layer_1_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(input_layer)
    relu_layer_1_1 = Activation('relu')(conv_layer_1_1)
    conv_layer_1_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(relu_layer_1_1)
    relu_layer_1_2 = Activation('relu')(conv_layer_1_2)
    pooling_layer_1 = MaxPooling1D(pool_size = pool_size, padding='same')(relu_layer_1_2)
    dropout_layer_1_1 = Dropout(rate=dropout_rate)(relu_layer_1_1)
    flattened_layer = Flatten()(pooling_layer_1)
    dropout_flattened = Dropout(rate=dropout_rate)(flattened_layer)
    output_layer = Dense(units=nb_classes, activation='softmax')(dropout_flattened)
    model = Model(inputs=input_layer, outputs=output_layer)
    optimizer_algo = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=cost_function,optimizer=optimizer_algo, metrics=['accuracy'])
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best-model.keras', monitor='val_loss', save_best_only=True)
    history = model.fit(X_train_normalized, y_train_encoded, 
                    batch_size=mini_batch_size, 
                    epochs=nb_epochs,
                    validation_split=percentage_of_train_as_validation,
                    verbose=False,
                    callbacks=[model_checkpoint])
    
    # Évaluer la performance du meilleur modèle
    best_model = tf.keras.models.load_model('best-model.keras')
    test_loss, test_accuracy = best_model.evaluate(X_train_normalized, y_test_encoded)
    
    # Mettre à jour les meilleurs hyperparamètres si nécessaire
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_loss = test_loss
        best_hyperparameters = (filters, kernel_size, stride, pool_size, dropout_rate, learning_rate, mini_batch_size)

# Fin du compteur de temps
end_time = time.time()

# Calcul de la durée d'entraînement en secondes
training_time_seconds = end_time - start_time

print("Meilleurs hyperparamètres :", best_hyperparameters)
print("Meilleure perte :", best_loss)
print("Meilleure précision :", best_accuracy)
print(f"Temps d'entraînement des hyperparamètre : {training_time_seconds // 60} minutes et {training_time_seconds % 60} secondes.")


# %%
"""# input
input_shape = (96, 1)
input_layer = Input(input_shape)

# block 1
conv_layer_1_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(input_layer)
relu_layer_1_1 = Activation('relu')(conv_layer_1_1)
conv_layer_1_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(relu_layer_1_1)
relu_layer_1_2 = Activation('relu')(conv_layer_1_2)
pooling_layer_1 = MaxPooling1D(pool_size = pool_size, padding='same')(relu_layer_1_2)
dropout_layer_1_1 = Dropout(rate=dropout_rate)(relu_layer_1_1)

# block 2
conv_layer_2_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(pooling_layer_1)
relu_layer_2_1 = Activation('relu')(conv_layer_2_1)
conv_layer_2_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(relu_layer_2_1)
relu_layer_2_2 = Activation('relu')(conv_layer_2_2)
pooling_layer_2 = MaxPooling1D(pool_size = pool_size, padding='same')(relu_layer_2_2)

# block 3
conv_layer_3_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(pooling_layer_2)
relu_layer_3_1 = Activation('relu')(conv_layer_3_1)
conv_layer_3_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(relu_layer_3_1)
relu_layer_3_2 = Activation('relu')(conv_layer_3_2)
pooling_layer_3 = MaxPooling1D(pool_size = pool_size, padding='same')(relu_layer_3_2)

# block 4
conv_layer_4_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(pooling_layer_3)
relu_layer_4_1 = Activation('relu')(conv_layer_4_1)
conv_layer_4_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding='same')(relu_layer_4_1)
relu_layer_4_2 = Activation('relu')(conv_layer_4_2)
pooling_layer_4 = MaxPooling1D(pool_size = pool_size, padding='same')(relu_layer_4_2)

# output
flattened_layer = Flatten()(pooling_layer_1)
dropout_flattened = Dropout(rate=dropout_rate)(flattened_layer)
output_layer = Dense(units=nb_classes, activation='softmax')(dropout_flattened)

model = Model(inputs=input_layer, outputs=output_layer)
model.compile(loss=cost_function,optimizer=optimizer_algo, metrics=['accuracy'])

tf.keras.utils.plot_model(model, show_shapes=True)

# specify the model checkpoint (to save the best model for each epoch)
model_checkpoint = tf.keras.callbacks.ModelCheckpoint('best-model.h5', monitor='val_loss', save_best_only=True)

# Début du compteur de temps
start_time = time.time()

# start training
history = model.fit(X_train_normalized, y_train_encoded, 
                    batch_size=mini_batch_size, 
                    epochs=nb_epochs,
                    validation_split=percentage_of_train_as_validation,
                    verbose=False,
                    callbacks=[model_checkpoint])

# Fin du compteur de temps
end_time = time.time()

# Calcul de la durée d'entraînement en secondes
training_time_seconds = end_time - start_time

# Affichage de la durée d'entraînement en minutes et secondes
print(f"Temps d'entraînement : {training_time_seconds // 60} minutes et {training_time_seconds % 60} secondes.")

"""



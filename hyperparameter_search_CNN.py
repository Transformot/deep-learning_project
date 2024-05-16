import numpy as np

from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.layers import Input, Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import models, callbacks

import time
import random

train_data = np.loadtxt('data/ECG200_TRAIN.tsv', delimiter='\t')
test_data = np.loadtxt('data/ECG200_TEST.tsv', delimiter='\t')

train_data[train_data[:, 0] == -1, 0] = 0
test_data[test_data[:, 0] == -1, 0] = 0
X_train, y_train = train_data[:, 1:], train_data[:, 0]
X_test, y_test = test_data[:, 1:], test_data[:, 0]
X_train_normalized = normalize(X_train, axis=1)
X_test_normalized = normalize(X_test, axis=1)
y_train_encoded = to_categorical(y_train, num_classes=2)
y_test_encoded = to_categorical(y_test, num_classes=2)

# hyperparamètres
filters = 5
kernel_size = 3
stride = 1
padding = 'same'
use_bias = False
hidden_activation = 'relu'
final_activation = 'sigmoid' 
pool_size = 2
dropout_rate = 0.2
nb_classes = 2
# learning_rate = 0.006
optimizer_algo = 'adam'  # Adam(learning_rate=learning_rate)
cost_function = 'binary_crossentropy'
mini_batch_size = 16
nb_epochs = 800
percentage_of_train_as_validation = 0.2

# Définir les plages de valeurs pour les hyperparamètres
filters_range = [5]
kernel_size_range = [3]                        # [1, 2, 3]
strides_range = [1]                          # [1, 2, 3]
pool_size_range = [2]                                # [1, 2, 3]
dropout_rate_range = [0.2]
mini_batch_size_range = [16, 32]

# Nombre total de combinaisons à essayer
total_combinations = len(filters_range) * len(kernel_size_range) * len(strides_range) \
                   * len(pool_size_range) \
                   * len(dropout_rate_range) * len(mini_batch_size_range)
print(total_combinations)

# Nombre de combinaisons à essayer
num_trials = 60*1

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
    pool_size = random.choice(pool_size_range)
    dropout_rate = random.choice(dropout_rate_range)
    mini_batch_size = random.choice(mini_batch_size_range)
    
    # build and compil model
    input_shape = (96, 1)
    input_layer = Input(input_shape)
    conv_layer_1_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(input_layer)
    relu_layer_1_1 = Activation(hidden_activation)(conv_layer_1_1)
    conv_layer_1_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(relu_layer_1_1)
    relu_layer_1_2 = Activation(hidden_activation)(conv_layer_1_2)
    pooling_layer_1 = MaxPooling1D(pool_size=pool_size, padding=padding)(relu_layer_1_2)
    dropout_layer_1_1 = Dropout(rate=dropout_rate)(pooling_layer_1)
    conv_layer_2_1 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(dropout_layer_1_1)
    relu_layer_2_1 = Activation(hidden_activation)(conv_layer_2_1)
    conv_layer_2_2 = Conv1D(filters=filters,kernel_size=kernel_size,strides=stride,padding=padding,use_bias=use_bias)(relu_layer_2_1)
    relu_layer_2_2 = Activation(hidden_activation)(conv_layer_2_2)
    pooling_layer_2 = MaxPooling1D(pool_size=pool_size, padding=padding)(relu_layer_2_2)
    flattened_layer = Flatten()(pooling_layer_2)
    dropout_flattened = Dropout(rate=dropout_rate)(flattened_layer)
    output_layer = Dense(units=nb_classes,activation=final_activation)(dropout_flattened)
    model = models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['accuracy'])
    
    model_checkpoint = callbacks.ModelCheckpoint('best_model_CNN.keras', monitor='val_loss', save_best_only=True)

    # start training
    history = model.fit(X_train_normalized, y_train_encoded, 
                        batch_size=mini_batch_size, 
                        epochs=nb_epochs,
                        validation_split=percentage_of_train_as_validation,
                        verbose=False,
                        callbacks=[model_checkpoint])

    # Eveluate best model
    best_model = models.load_model('best_model_CNN.keras')
    train_loss, train_accuracy = best_model.evaluate(X_train_normalized, y_train_encoded)
    test_loss, test_accuracy = best_model.evaluate(X_test_normalized, y_test_encoded)
    print((filters, kernel_size, stride, pool_size, dropout_rate, mini_batch_size))
    
    # Mettre à jour les meilleurs hyperparamètres si nécessaire
    if test_accuracy - test_loss > best_accuracy - best_loss :
        best_accuracy = test_accuracy
        best_loss = test_loss
        best_hyperparameters.append((filters, kernel_size, stride, pool_size, dropout_rate, mini_batch_size))
    if test_accuracy - test_loss > 0.47 :
        best_hyperparameters_2.append((filters, kernel_size, stride, pool_size, dropout_rate, mini_batch_size))

# Fin du compteur de temps
end_time = time.time()

# Calcul de la durée d'entraînement en secondes
training_time_seconds = end_time - start_time

print("Best hyperparameters:", best_hyperparameters)
print("Best hyperparameters 2:", best_hyperparameters_2)
print("Meilleure perte :", best_loss)
print("Meilleure précision :", best_accuracy)
print(f"Temps d'entraînement des hyperparamètre : {training_time_seconds // 60} minutes et {training_time_seconds % 60} secondes.")

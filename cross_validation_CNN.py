import numpy as np

from tensorflow.keras.utils import normalize, to_categorical
from tensorflow.keras.layers import Input, Conv1D, Activation, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras import models, callbacks

train_data = np.loadtxt('data/ECG200_TRAIN.tsv', delimiter='\t')
test_data = np.loadtxt('data/ECG200_TEST.tsv', delimiter='\t')

test_data[test_data[:, 0] == -1, 0] = 0
X_test, y_test = test_data[:, 1:], test_data[:, 0]
X_test_normalized = normalize(X_test, axis=1)
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

# Création de 5 ensembles de validation croisée car 20% de validation
num_splits = 5
test_accuracy_vals = []
test_loss_vals = []
train_accuracy_vals = []
train_loss_vals = []

# Mélanger les données
for i in range(num_splits) :
    train_data_suffled = train_data
    np.random.shuffle(train_data_suffled)
    train_data_suffled[train_data_suffled[:, 0] == -1, 0] = 0
    X_train, y_train = train_data_suffled[:, 1:], train_data_suffled[:, 0]
    X_train_normalized = normalize(X_train, axis=1)
    y_train_encoded = to_categorical(y_train, num_classes=2)

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

    test_accuracy_vals.append(test_accuracy)
    test_loss_vals.append(test_loss)
    train_accuracy_vals.append(train_accuracy)
    train_loss_vals.append(train_loss)

# Affichage des moyennes
print("Moyenne de test_accuracy_vals:", np.mean(test_accuracy_vals))
print("Moyenne de test_loss_vals:", np.mean(test_loss_vals))
print("Moyenne de train_accuracy_vals:", np.mean(train_accuracy_vals))
print("Moyenne de train_loss_vals:", np.mean(train_loss_vals))
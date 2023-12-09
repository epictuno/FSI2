from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

import os
import scipy
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import random
import numpy as np
import tensorflow as tf
from keras.callbacks import EarlyStopping
from keras.layers import Dense
from keras.metrics import Precision, Recall
# Establecer una semilla para reproducibilidad
seed_value = 42
random.seed(seed_value)        # Semilla para Python
np.random.seed(seed_value)     # Semilla para NumPy
tf.random.set_seed(seed_value)  # Semilla para TensorFlow/Keras





def generatePlot(history):
    # Extracción de datos
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    # Creación del gráfico de accuracy
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training')
    plt.plot(epochs, val_acc, 'r', label='Validation')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Creación del gráfico de loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training')
    plt.plot(epochs, val_loss, 'r', label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

def generador_de_imagenes( directory ):
    for filename in os.listdir(directory):
        # Cargar la imagen
        img = load_img(os.path.join(directory, filename))
        x = img_to_array(img)  # Convertir la imagen a un array Numpy
        x = x.reshape((1,) + x.shape)  # Remodelar para tener 4 dimensiones
        # Generar imágenes
        i = 0
        for batch in train_datagen.flow(x, batch_size=1, save_to_dir=directory, save_prefix='img', save_format='jpeg'):
            i += 1
            if i > 5:  # Ajusta este número para generar más o menos imágenes
                break  # Rompe el bucle, de lo contrario generaría imágenes indefinidamente




image_size = 240
batch_size = 16
rescale_factor = 1. / 255

# Crear un generador para aumentar datos
train_datagen = ImageDataGenerator(
    rescale=rescale_factor,  # Normalizar los valores de los píxeles
    shear_range=0.2,  # Rango para las transformaciones aleatorias
    zoom_range=0.2,  # Rango para el zoom aleatorio
    horizontal_flip=True,  # Activar el giro horizontal aleatorio
    validation_split=0.1)  # Establecer el porcentaje de imágenes para el conjunto de validación
#generador_de_imagenes("../pythonProject/archive/train/adidas")
#generador_de_imagenes("../pythonProject/archive/train/nike")
#generador_de_imagenes("../pythonProject/archive/train/converse")
# Cargar imágenes de entrenamiento
train_generator = train_datagen.flow_from_directory(
    "../pythonProject/archive/train",  # Directorio con datos
    target_size=(image_size, image_size),  # Cambiar el tamaño de las imágenes a 50x50
    batch_size=batch_size,
    shuffle=True,
    class_mode='categorical',  # 'binary' para clasificación binaria, 'categorical' para multiclase
    subset='training')  # Seleccionar solo el conjunto de entrenamiento

# Cargar imágenes de validación
validation_generator = train_datagen.flow_from_directory(
    "../pythonProject/archive/train",
    target_size=(image_size, image_size),
    shuffle= True,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')  # Seleccionar solo el conjunto de validación

model = Sequential()
# Capas convolucionales
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(240, 240, 3)))  # 3 canales de color
model.add(Dropout(0.25))  # Dropout después de la capa de conv2D
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(4, 4)))
#model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
#model.add(Dropout(0.5))
#model.add(MaxPooling2D(pool_size=(4, 4)))



model.add(Flatten())  # Aplanar la salida de la capa convolucional
# Capas fully connected (clasificador)
model.add(Dense(128, activation='relu'))

model.add(Dense(64, activation='relu'))

#model.add(Dense(32, activation='relu'))

model.add(Dropout(0.5))

# Dropout antes de la capa de salida
model.add(Dense(3, activation='softmax'))
print(model.summary())


# Configurar Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)  # 'patience' es el número de épocas sin mejora después de las cuales el entrenamiento se detendrá
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(), Recall()])
epochs = 50

# Entrenar el modelo con Early Stopping
history_of_train = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=[early_stopping]
)

generatePlot(history_of_train)

# Evaluar el modelo en el conjunto de validación (usado aquí como prueba)
# Puedes limitar el número de pasos para usar solo una parte del conjunto
test_loss, test_accuracy, precision, recall = model.evaluate(validation_generator, steps=50)  # 'steps' es el número de lotes a evaluar
print(f'Precisión: {precision}, Sensibilidad: {recall}')
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
#model.save('my_model3.h5')




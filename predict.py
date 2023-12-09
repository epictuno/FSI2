import os

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix
from PIL import Image

# Carga el modelo entrenado
model = load_model('my_model3.h5')
"""
img_dir = 'C:/Users/Usuario/Desktop/practicas ap/pythonProject/archive/test/adidas/18.jpg'
img = image.load_img(img_dir, target_size=(240, 240))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

# Usa el modelo para predecir el tipo de la imagen
preds = model.predict(x)

# Imprime las predicciones
print('Predicciones:', preds)

# Directorio de las imágenes
img_dir = 'C:/Users/Usuario/Desktop/practicas ap/pythonProject/archive/test/adidas'

# Obtiene una lista de todas las imágenes en el directorio
all_images = os.listdir(img_dir)

# Selecciona sólo las primeras 10 imágenes
first_10_images = all_images[:50]

# Recorre las primeras 10 imágenes en el directorio
for img_name in first_10_images:
    # Carga y preprocesa la imagen
    img_path = os.path.join(img_dir, img_name)
    img = image.load_img(img_path, target_size=(240, 240))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Usa el modelo para predecir el tipo de la imagen
    preds = model.predict(x)

    # Imprime las predicciones
    print(f'Predicciones para {img_name}:', preds)
"""

def cargar_imagenes(directorio):
    imagenes = []
    for archivo in os.listdir(directorio):
        if archivo.endswith('.jpg'):  # Asegúrate de que el formato sea correcto
            img = Image.open(os.path.join(directorio, archivo))
            #img = img.resize((tamaño_deseado, tamaño_deseado))  # Cambia el tamaño si es necesario
            img_array = np.array(img)
            imagenes.append(img_array)
    return np.array(imagenes)

# Carga las imágenes de cada clase
x_test_a = cargar_imagenes('../pythonProject/archive/test/adidas')
x_test_b = cargar_imagenes('../pythonProject/archive/test/converse')
x_test_c = cargar_imagenes('../pythonProject/archive/test/nike')
x_test = np.concatenate((x_test_a, x_test_b, x_test_c), axis=0)
x_test = x_test / 255.0
y_test = np.array([0]*len(x_test_a) + [1]*len(x_test_b) + [2]*len(x_test_c))


y_pred = model.predict(x_test)
y_pred = np.argmax(y_pred, axis=1)
cm = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred, normalize='pred')
class_names = ['Adidas', 'Converse', 'Nike']
cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)

# Creamos el mapa de calor utilizando Seaborn
plt.figure(figsize=(8,6))
sns.heatmap(cm_df, annot=True, fmt='g', cmap='Blues')
plt.title('Matriz de Confusión')
plt.ylabel('Valores Reales')
plt.xlabel('Valores Predichos')
plt.show()

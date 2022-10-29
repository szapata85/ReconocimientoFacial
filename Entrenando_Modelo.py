#------------------Importar las librerias-----------------
import cv2
import numpy as np
import os

#----------Importar fotos tomadas anteriormente
direccion = r'C:\Users\CHECHO\Desktop\Deteccion_Rostros\Clase2\Fotos'
lista = os.listdir(direccion)

etiquetas = []
rostros = []
count = 0

for nameDir in lista:
    nombre = direccion + '/' + nameDir #leer las fotos de los rostros

    for fileName in os.listdir(nombre):
        etiquetas.append(count)
        rostros.append(cv2.imread(nombre + '/' + fileName,0))

    count = count + 1

#--------------Creamos el modelo--------------------------
reconocimiento = cv2.face.LBPHFaceRecognizer_create()

#--------------Entrenamos el modelo-----------------------
reconocimiento.train(rostros, np.array(etiquetas))

#--------------Guardamos el Modelo------------------------
reconocimiento.write('ModeloEntrenado.xml')
print('Modelo Creado')
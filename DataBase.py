#------------------------Importamos Librerias--------------------------
import cv2
import mediapipe as mp
import os

#------------------------Creacion de la carpeta donde se almacena las fotos--------------------------
nombre = 'Checho_sin_Tapabocas'
direccion = r'C:\Users\CHECHO\Desktop\Deteccion_Rostros\Clase2\Fotos'
carpeta = direccion + '/' + nombre

if not os.path.exists(carpeta):
    print("Carpeta Creada")
    os.makedirs(carpeta)

#-------------inicializamos nuestro contados------------------
count = 0

#-------------declaramos el detector---------------------------
detector = mp.solutions.face_detection #Detector
dibujo = mp.solutions.drawing_utils #Herramienta de Dibujo

#------------Realizar la Video Captura -----------------------
cap = cv2.VideoCapture(0)

#------------Inicializamos parametros de la deteccion---------
with detector.FaceDetection(min_detection_confidence=0.75) as rostros:

    #Inicializamos While True
    while True:
        # Realizamos la lectura de la VideoCaptura
        ret, frame = cap.read()

        # Eliminar error de movimiento para invertir la captura de la camara
        frame = cv2.flip(frame, 1)

        # Correccion de color
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # deteccion de rostros
        resultado = rostros.process(rgb)

        # Filtro de seguridad
        if resultado.detections is not None:
            for rostro in resultado.detections:
                #dibujo.draw_detection(frame, rostro, dibujo.DrawingSpec(color=(0, 255, 0)))

                for id, coordenadas in enumerate(resultado.detections):
                    # Mostramos las coordenadas
                    # print("Coordenadas: ", coordenadas)

                    # conversion de coordenadas
                    al, an, c = frame.shape

                    # Extraer X inicial e Y inicial
                    xi = coordenadas.location_data.relative_bounding_box.xmin
                    yi = coordenadas.location_data.relative_bounding_box.ymin

                    # Extraer ancho y alto
                    ancho = coordenadas.location_data.relative_bounding_box.width
                    alto = coordenadas.location_data.relative_bounding_box.height

                    # conversion a pixeles
                    xi, yi = int(xi * an), int(yi * al)
                    ancho, alto = int(ancho * an), int(alto * al)

                    #Hallamos Xfinal e YFinal
                    xf = xi + ancho
                    yf = yi + alto

                    #Extraccion de pixeles
                    cara = frame[yi:yf, xi:xf]

                    #redimencionar las fotos
                    cara = cv2.resize(cara, (150,200), interpolation=cv2.INTER_CUBIC)

                    #Almacenar Nuestras Imagenes
                    cv2.imwrite(carpeta + '/rostro_{}.jpg'.format(count),cara)
                    count = count + 1


        #Mostramos los fotogramas
        cv2.imshow("Reconocimiento Facial y de Tapabocas", cara)

        # leemos el teclado
        t = cv2.waitKey(1)
        if t == 27 or count > 300:
            break

cap.release()
cv2.destroyAllWindows()
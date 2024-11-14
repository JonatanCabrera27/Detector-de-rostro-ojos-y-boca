import cv2

# Cargar los clasificadores de Haar para la detección de rostros y ojos
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_mouth.xml')
nose_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_mcs_nose.xml')

# Iniciar la captura de video desde la cámara web
cap = cv2.VideoCapture(0)

# Bucle para procesar cada cuadro en tiempo real
while True:
    # Capturar cuadro por cuadro
    ret, frame = cap.read()
    if not ret:
        break

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Dibujar un rectángulo alrededor de cada rostro detectado y detectar ojos en cada rostro
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extraer la región del rostro para la detección de ojos
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        # Detectar ojos en la región del rostro
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(15, 15))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Detectar la nariz en la región del rostro (debajo de los ojos)
        noses = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(20, 20))
        for (nx, ny, nw, nh) in noses:
            # Asegurarse de que la nariz esté en la parte media del rostro
            cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (0, 0, 255), 2)


        mouths = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=10, minSize=(30, 30))
        for (mx, my, mw, mh) in mouths:
            # Asegurarse de que la boca esté en la parte inferior del rostro
            if my > h / 2:  # Asegura que la boca esté debajo de la mitad del rostro
                cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 50, 0), 2)


    # Mostrar el cuadro con las detecciones
    cv2.imshow('Detector de Rostros y Ojos', frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar los recursos
cap.release()
cv2.destroyAllWindows()

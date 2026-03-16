import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH = "Video/videoTest.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Error: no se pudo abrir el video.")
    exit()

# Obtenemos los FPS del video
FPS = cap.get(cv2.CAP_PROP_FPS)

# Sustraemos el fondo utilizando MOG2
back_sub = cv2.createBackgroundSubtractorMOG2(
    history=200,
    varThreshold=50,
    detectShadows=False
)

# Matriz para poder hacer las operaciones morfológicas
kernel = np.ones((5, 5), np.uint8)

AREA_MINIMA = 500

historial_centroides = []

# VARIABLES PARA ANALISIS FISICO
tiempos = []
posiciones_px = []
velocidad = 0

# Puntos A y B usados para la escala métrica
punto_A = (80, 450)
punto_B = (1200, 450)
mostrar_escala = True

# Escala pixel → metros
PIXEL_A_METRO = 4.5 / 751

# Definimos vistas para tener un mejor control sobre los diferentes procesos que hemos hecho
VISTAS = [
    "1. Original",
    "2. Escala de grises",
    "3. Mascara cruda",
    "4. Erosion",
    "5. Dilatacion",
    "6. Apertura",
    "7. Cierre",
    "8. Mascara limpia",
    "9. Contornos detectados",
    "10. Centroide y trayectoria",
]
vista_actual = 0

cv2.namedWindow("Segmentacion", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Segmentacion", 1280, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_actual = cap.get(cv2.CAP_PROP_POS_FRAMES)
    segundo_actual = frame_actual / FPS

    # PREPROCESAMIENTO
    # Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Sustracciónde fondo
    mask = back_sub.apply(gray)

    # Operaciones morfológicas
    erosion = cv2.erode(mask, kernel, iterations=1)
    dilatacion = cv2.dilate(mask, kernel, iterations=1)
    apertura = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    cierre = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_limpia = cv2.morphologyEx(apertura, cv2.MORPH_CLOSE, kernel)

    # Contornos del vehículo
    contornos, _ = cv2.findContours(
        mask_limpia,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contorno_vehiculo = None

    if contornos:
        contorno_mas_grande = max(contornos, key=cv2.contourArea)
        if cv2.contourArea(contorno_mas_grande) > AREA_MINIMA:
            contorno_vehiculo = contorno_mas_grande

    # Calculamos centroide con momentos
    cx, cy = None, None
    if contorno_vehiculo is not None:
        M = cv2.moments(contorno_vehiculo)
        if M["m00"] != 0:  # evitar división por cero
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            historial_centroides.append((cx, cy))

            # Calculo de velocidad por fotograma
            if len(historial_centroides) >= 2:
                x1, y1 = historial_centroides[-2]
                x2, y2 = historial_centroides[-1]
                distancia_pixeles = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                distancia_metros = distancia_pixeles * PIXEL_A_METRO
                tiempo = 1 / FPS
                # Velocidad instantanea por fotograma
                velocidad = distancia_metros / tiempo

            # Guardamos datos para análisis posterior
            tiempos.append(segundo_actual)
            posiciones_px.append(cx)

    # Dibujamos el contorno y lo delimitamos
    frame_contornos = frame.copy()
    if contorno_vehiculo is not None:
        cv2.drawContours(frame_contornos, [contorno_vehiculo], -1, (0,255,0), 2)
        x, y, w, h = cv2.boundingRect(contorno_vehiculo)
        cv2.rectangle(frame_contornos,(x,y),(x+w,y+h),(255,0,0),2)

    # Dibujamos la trayectoria y el centroide
    frame_centroide = frame.copy()
    for i in range(1, len(historial_centroides)):
        cv2.line(frame_centroide,
                historial_centroides[i - 1],
                historial_centroides[i],
                (0,255,255), 2)   # línea amarilla

    if cx is not None and cy is not None:
        cv2.circle(frame_centroide,(cx,cy),6,(0,0,255),-1)
        cv2.putText(frame_centroide, f"({cx}, {cy})",
                    (cx+10,cy-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0, 0, 255), 2)
    
    # Seleccionar display
    if vista_actual == 0:
        display = frame.copy()
    elif vista_actual == 1:
        display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 2:
        display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 3:
        display = cv2.cvtColor(erosion, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 4:
        display = cv2.cvtColor(dilatacion, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 5:
        display = cv2.cvtColor(apertura, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 6:
        display = cv2.cvtColor(cierre, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 7:
        display = cv2.cvtColor(mask_limpia, cv2.COLOR_GRAY2BGR)
    elif vista_actual == 8:
        display = frame_contornos
    elif vista_actual == 9:
        display = frame_centroide

    # HUD
    cv2.putText(display,VISTAS[vista_actual],(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1.2,(0,255,0),2)
    cv2.putText(display,f"Velocidad: {velocidad:.2f} m/s",(20,120),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(display,f"Tiempo: {segundo_actual:.2f} s",(20,80),
                cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2)
    cv2.putText(display,"[a] Anterior   [d] Siguiente   [s] Escala Metrica   [q] Salir",
                (20, display.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)

    # Dibujar puntos A y B
    cv2.circle(display,punto_A,8,(255,0,0),-1)
    cv2.circle(display,punto_B,8,(0,0,255),-1)
    cv2.putText(display,"A",(punto_A[0]+10,punto_A[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),2)
    cv2.putText(display,"B",(punto_B[0]+10,punto_B[1]-10),
                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)
    
    # Dibujamos la linea de escala entre el punto A y B
    if mostrar_escala:
        cv2.line(display, punto_A, punto_B, (0,0,0), 2)
        cv2.putText(display, "7 m",
                    ((punto_A[0]+punto_B[0])//2, (punto_A[1]+punto_B[1])//2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2)

    cv2.imshow("Segmentacion", display)

    # Manejo de keybinds
    key = cv2.waitKey(30) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("a"):
        vista_actual = (vista_actual - 1) % len(VISTAS)
    elif key == ord("d"):
        vista_actual = (vista_actual + 1) % len(VISTAS)
    elif key == ord('s'):
        mostrar_escala = not mostrar_escala

cap.release()
cv2.destroyAllWindows()

# -------------------------------
# ANALISIS DEL MOVIMIENTO
# -------------------------------

# Función que suaviza una lista de datos usando un promedio móvil
def suavizar(datos, ventana=5):
    datos_suavizados = [] 

    for i in range(len(datos)):
        inicio = max(0, i - ventana//2)
        fin = min(len(datos), i + ventana//2 + 1)

        promedio = sum(datos[inicio:fin]) / (fin - inicio)
        datos_suavizados.append(promedio)

    return datos_suavizados

posiciones_m = [p * PIXEL_A_METRO for p in posiciones_px]
posiciones_m = suavizar(posiciones_m, ventana=7)

velocidades = []

# Calcula la velocidad entre cada par de posiciones consecutivas
for i in range(len(posiciones_m) - 1):

    dx = posiciones_m[i + 1] - posiciones_m[i]
    dt = tiempos[i + 1] - tiempos[i]

    # Evita división por cero
    if dt != 0:
        velocidades.append(dx / dt)

velocidades = suavizar(velocidades, ventana=5)
aceleraciones = []

# Calcula la aceleración entre cada par de velocidades consecutivas
for i in range(len(velocidades) - 1):

    dv = velocidades[i + 1] - velocidades[i]
    dt = tiempos[i + 1] - tiempos[i]

    if dt != 0:
        aceleraciones.append(dv / dt)

aceleraciones = suavizar(aceleraciones, ventana=5)

# Crea una nueva figura para los gráficos
plt.figure()

# Primer gráfico: Posición vs Tiempo
plt.subplot(3, 1, 1)
plt.plot(tiempos, posiciones_m)
plt.title("Posicion vs Tiempo")
plt.ylabel("Posicion (m)")

# Segundo gráfico: Velocidad vs Tiempo
plt.subplot(3, 1, 2)
plt.plot(tiempos[:-1], velocidades)
plt.title("Velocidad vs Tiempo")
plt.ylabel("Velocidad (m/s)")

# Tercer gráfico: Aceleración vs Tiempo
plt.subplot(3, 1, 3)
plt.plot(tiempos[:-2], aceleraciones)
plt.title("Aceleracion vs Tiempo")
plt.ylabel("Aceleracion (m/s²)")
plt.xlabel("Tiempo (s)")

plt.tight_layout()
plt.show()

# Imprime cuántos datos de posición se registraron
print("Datos guardados:", len(posiciones_px))
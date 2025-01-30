import cv2
import numpy as np
import easyocr
import os
import matplotlib.pyplot as plt
plt.close('all')

# citirea imaginii
img = cv2.imread('imagine_4.jpg')

# detetectie placuta
license_plate, detected_text = detect_license_plate_and_read_text(img)

# afisare contur placuta
cv2.drawContours(img, [license_plate], -1, (0, 255, 0), 3)
        
# afisare text detectat
cv2.putText(img, detected_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def detect_license_plate_and_read_text(img):

    # redimensionare pentru preprocesare mai usoara si standardizare set de date
    img = cv2.resize(img, (800, 600))
    
    # conversie format grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # filtru gaussian pentru reducere zgomot
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # detectie contur cu Canny
    edges = cv2.Canny(blurred, 100, 200)
    
    # identificarea contururilor
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # ordonarea contururilor in functie de aria lor (descrescator primele 10)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate = None
    for contour in contours:
        # se aproximeaza forma contururilor
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        
        # selectez doar contururile aproximate cu 4 laturi (posibile placute de inmatriculare)
        if len(approx) == 4:
            license_plate = approx
            break
    
    detected_text = None
    if license_plate is not None:
        # creare masca pentru zona detectata
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [license_plate], -1, 255, -1)
        
        # se extrage doar zona placutei folosind masca
        x, y, w, h = cv2.boundingRect(license_plate)
        cropped_plate = gray[y:y+h, x:x+w]
        
        # recunoasterea textului din zona de interes folosind EasyOcr
        reader = easyocr.Reader(['en'], gpu=False)
        result = reader.readtext(cropped_plate)
        
        detected_text = " ".join([res[1] for res in result])

    
    else:
        print(f"Nicio placuta de inmatriculare nu a fost detectata in {image_path}.")


    return license_plate, detected_text



















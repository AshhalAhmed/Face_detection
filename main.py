import cv2

a = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
b = cv2.VideoCapture(0)

while True:
    ret, img = b.read()
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)  # Rotate the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = a.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 5)
        cv2.putText(img, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
        break

b.release()
cv2.destroyAllWindows()

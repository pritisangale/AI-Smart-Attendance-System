import cv2
import os
# Ask student name
student_name = input("Enter student name: ")

# Create folder for student
dataset_path = "dataset/student_faces/" + student_name
os.makedirs(dataset_path, exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Start webcam
cap = cv2.VideoCapture(0)

count = 0

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:

        count += 1

        face = frame[y:y+h, x:x+w]

        file_name = dataset_path + "/" + str(count) + ".jpg"

        cv2.imwrite(file_name, face)

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)

    cv2.imshow("Capturing Faces", frame)

    # Stop after 50 images
    if count >= 50:
        break

    # Press q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection completed")

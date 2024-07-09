import cv2
import os
import numpy as np
import tkinter as tk
import tkinter.font as font
global labels
def collect_data():
    name = input("Enter name of person: ")
    count = 1
    ids = input("Enter ID: ")

    cap = cv2.VideoCapture(0)
    filename = r"C:\Users\Thasmai\Downloads\safehome (1)\haarcascade_frontalcatface.xml"
    cascade = cv2.CascadeClassifier(filename)

    while True:
        ret, frm = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.4, 1)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]

            cv2.imwrite(f"persons/{name}-{count}-{ids}.jpg", roi)
            count += 1
            cv2.putText(frm, f"{count}", (20, 20), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

        cv2.imshow("Collect Data", frm)

        if cv2.waitKey(1) == 27 or count > 300:
            cv2.destroyAllWindows()
            cap.release()
            train()
            break
def train():
    print("Training initiated!")
    #globel labels
    recog = cv2.face.LBPHFaceRecognizer_create()
    dataset = 'persons'
    paths = [os.path.join(dataset, im) for im in os.listdir(dataset)]

    faces = []
    ids = []
    labels={}

    for path in paths:
        label = path.split('/')[-1].split('-')[0]
        ids.append(int(path.split('/')[-1].split('-')[2].split('.')[0]))
        faces.append(cv2.imread(path, 0))
        labels[ids[-1]] = label

    recog.train(faces, np.array(ids))
    recog.save('model.yml')


def identify():
    cap = cv2.VideoCapture(0)
    cascade = cv2.CascadeClassifier(r"C:\Users\Thasmai\Downloads\safehome (1)\haarcascade_frontalcatface.xml")
    recog = cv2.face.LBPHFaceRecognizer_create()
    recog.read('model.yml')

    # Define your dictionary mapping IDs to names
    labels_mapping = { 1 : "thasmai"}

    while True:
        ret, frm = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.3, 2)

        for x, y, w, h in faces:
            cv2.rectangle(frm, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi = gray[y:y+h, x:x+w]
            label_id, confidence = recog.predict(roi)

            # Check if the confidence level is below a certain threshold
            if confidence < 100:
                name = labels_mapping.get(label_id, "Unknown")
                cv2.putText(frm, f"Name: {name}, ID: {label_id}, Confidence: {int(confidence)}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frm, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow("Identify", frm)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


def maincall():
    root = tk.Tk()
    root.geometry("480x100")
    root.title("Face Recognition")

    label = tk.Label(root, text="Select an option below:")
    label.grid(row=0, columnspan=2)
    label_font = font.Font(size=14, weight='bold')
    label['font'] = label_font

    button1 = tk.Button(root, text="Add Member", command=collect_data)
    button1.grid(row=1, column=0, pady=(10, 10), padx=(5, 5))

    button2 = tk.Button(root, text="Start Recognition", command=identify)
    button2.grid(row=1, column=1, pady=(10, 10), padx=(5, 5))

    root.mainloop()

if __name__ == "main":
    maincall()
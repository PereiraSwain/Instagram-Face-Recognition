import cv2
import os
from flask import Flask, request, render_template, redirect
import numpy as np

from sklearn.ensemble import RandomForestClassifier

import joblib



# Defining Flask App
app = Flask(__name__)

nimgs = 25


# firebaseConfig = {
#     'apiKey': "AIzaSyAZe4nIe5J4gnZkCg2Nq8H0UIjn8GBUm_E",
#     'authDomain': "instagram-face-recognition.firebaseapp.com",
#     'databaseURL': 'https://instagram-face-recognition.firebaseio.com',
#     'projectId': "instagram-face-recognition",
#     'storageBucket': "instagram-face-recognition.appspot.com",
#     'messagingSenderId': "646845225022",
#     'appId': "1:646845225022:web:d9c8d441853b002bf0996f",
#     'measurementId': "G-NRVKVTH69M"
#   }
#
# firebase=pyrebase.initialize_app(firebaseConfig)
# storage=firebase.storage()




# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them//////
# list_img=['img1.jpg','img2.jpg']
# for i in range(len(list_img)):
#     print(list_img[i])
#     storage.child(f'static/faces/img{i}').put(list_img[i])
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')


# get a number of total registered users//////
def totalreg ():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    # knn = KNeighborsClassifier(n_neighbors=5)
    # knn.fit(faces, labels)
    classifier = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
    classifier.fit(faces, labels)

    joblib.dump(classifier, 'static/face_recognition_model.pkl')





# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


## A function to get names and rol numbers of all users
def getallusers():
    userlist = os.listdir('static/faces')
    names = []
    rolls = []
    l = len(userlist)

    for i in userlist:
        name, roll = i.split('_')
        names.append(name)
        rolls.append(roll)

    return userlist, names, rolls, l


## A function to delete a user folder
def deletefolder(duser):
    pics = os.listdir(duser)
    for i in pics:
        os.remove(duser+'/'+i)
    os.rmdir(duser)


# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

################## ROUTING FUNCTIONS #########################

# Our main page
@app.route('/')
def home():
    # names, rolls, times, l = extract_attendance()
    return render_template('home.html', totalreg=totalreg())
    # (, names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## List users page
# @app.route('/listusers')
# def listusers():
#     userlist, names, rolls, l = getallusers()
#     return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


## Delete functionality
# @app.route('/deleteuser', methods=['GET'])
# def deleteuser():
#     duser = request.args.get('user')
#     deletefolder('static/faces/'+duser)
#
#     ## if all the face are deleted, delete the trained file...
#     if os.listdir('static/faces/')==[]:
#         os.remove('static/face_recognition_model.pkl')
#
#     try:
#         train_model()
#     except:
#         pass
#
#     userlist, names, rolls, l = getallusers()
#     return render_template('listusers.html', userlist=userlist, names=names, rolls=rolls, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    # names, rolls, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', mess='There is no trained model in the static folder.Please add a new face '
                                                'to continue.', totalreg=totalreg())
        # (, names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2, mess='There is no
        # trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(0)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x+w, y-40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y+h, x:x+w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            # add_attendance(identified_person)
            cv2.putText(frame, f'{identified_person}', (x+5, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('insta', frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    username = identified_person.split('-')[2]
    # print(username)

    return redirect(f"https://www.instagram.com/{username}/", code=302)

    # (, names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    instausername = request.form['instausername']
    userimagefolder = 'static/faces/'+newusername+'-'+str(newuserid)+'-'+(instausername)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i, j = 0, 0
    cap = cv2.VideoCapture(0)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            if j % 5 == 0:
                name = newusername+'_'+str(i)+'.jpg'
                cv2.imwrite(userimagefolder+'/'+name, frame[y:y+h, x:x+w])
                i += 1
            j += 1
        if j == nimgs*5:
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    # names, rolls, times, l = extract_attendance()
    return render_template('home.html',totalreg=totalreg())
    # (, names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday2)


# Our main function which runs the Flask App
if __name__ == '__main__':
    app.run(debug=True)

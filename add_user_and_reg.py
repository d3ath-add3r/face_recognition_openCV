# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 15:49:33 2022
Project : Face_Recognition_Multi_User
@author: Nimish
"""
import cv2
import numpy as np
import os
from os import listdir
from os.path import isfile, join

def face_reg():
    '''
    

    Returns
    -------
    TYPE
        DESCRIPTION. State of Face Detection

    '''
    # Load face recognition model
    # Download from https://gist.github.com/Learko/8f51e58ac0813cb695f3733926c77f52#file-haarcascade_frontalface_default-xml
    
    
    
    
    
    
    # Collecting face images of the user for training data for image recognition
    def add_user():
        '''
        To add users
    
        Returns
        -------
        users : TYPE
            DESCRIPTION. users image data
    
        '''
        T = int(input("Enter number of users : "))
        users=[]
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        
        # Function to extract faces from the image or frame
        def face_extractor(img):
            '''
            
        
            Parameters
            ----------
            img : TYPE
                DESCRIPTION.
        
            Returns
            -------
            cropped_face : TYPE
                DESCRIPTION. Cropped image only the face
            rect : TYPE
                DESCRIPTION. Rect for the detected face
        
            '''
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            
            if faces is ():
                return None
            
            for (x,y,w,h) in faces:
                cropped_face = img[y:y+h, x:x+w]
                rect = [(x, y), (x + w, y + h)]
                
            return cropped_face, rect
    
        for t in range(T):
            name = input("Enter username : ")
            name = str(name)
            users.append(name)
        
    
            cap = cv2.VideoCapture(0)
            count = 0
    
            while True:
                ret, frame = cap.read()
    
                if face_extractor(frame) is not None:
                    count +=1
                    face = cv2.resize(face_extractor(frame)[0], (200, 200))
                    rect = face_extractor(frame)[1]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    
                    file_name_path = 'faces/' + name + str(count) + '.jpg'
    
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(frame, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.rectangle(frame, rect[0], rect[1], (0, 255, 255), 2)
                    cv2.imshow('FACE CROPPER', frame)
                    
                else:
                    print('face not found')
    
                if cv2.waitKey(1) == 13 or count == 200:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("Collecting {} Data Complete !!!".format(name))
        
        print(" All {} users data collected".format(T))
        return users
    
    users = add_user()
    
    data_path = 'faces/'
    onlyfiles= [f for f in listdir(data_path) if isfile(join(data_path, f))]
    face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    conf = int(input("Input Threshold for detection: "))
    # Making data ready for training
    Train_Data, Labels = [], []

    for file in onlyfiles:
        image_path = data_path + file
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Train_Data.append(np.asarray(image, dtype=np.uint8))
        Label = ''.join(filter(str.isalpha,file.split('.')[-2]))
        index = users.index(Label)
        
        Labels.append(index)
        
    Labels = np.asarray(Labels, dtype=int)
    
    # Linear Binary Phase Histogram Classifier

    model = cv2.face.LBPHFaceRecognizer_create()
    
    model.train(np.asarray(Train_Data), np.asarray(Labels))
    print('Model Training Complete !!!')
    
    def face_detector(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, )
        if faces is ():
            return img, []
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 255), 2)
            roi = img[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            
        return img, roi


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        
        image, face = face_detector(frame)
        
        try:
            
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)
            
            if result[1] < 500:
                confidence = int(100*(1-(result[1]) / 300))
                display_string = str(confidence) + '% Confidence it is USER - ' + users[result[0]]
                
            cv2.putText(image, display_string, (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 120, 255), 2)
            
            if confidence > conf:
                cv2.putText(image, "Unlocked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 55, 255), 2)
                cv2.imshow("Face Cropper", image)
                
            else:
                cv2.putText( image, "Locked", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 55, 0), 2)
                cv2.imshow("Face Cropper", image)
            
            state = "Face Found"
        except:
            cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            
            cv2.imshow("Face Cropper", image)
            state = "Face Not Found"
            pass
        
        if cv2.waitKey(1)==13:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    return state
    
# Calling face_reg function
if __name__ == "__main__":
    face_reg()


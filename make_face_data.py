import os
import face_recognition
import cv2 as cv


for parent, dirnames, filenames in os.walk('static/images'):
    # print parent
    # print dirnames
    # print filenames
    for dirname in dirnames:
        for subParent, subDirName, subFilenames in os.walk(parent + '/' + dirname):
            for filename in subFilenames:
                path = subParent + '/' + filename
                image = cv.imread(path)
                faces = face_recognition.face_locations(image)
                for i, (top, right, bottom, left) in enumerate(faces):
                    temp = image[top:bottom, left:right]
                print(path)
                cv.imwrite(path, temp)

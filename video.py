import numpy as np 
import cv2 
import sys 
import sys
from mtcnn.mtcnn import MTCNN
detector = MTCNN()
sys.path.insert(0, "model/")
sys.path.insert(0, "reco/")
from model import load_frmodel
from update_database import get_name,update


def run():
    capture = cv2.VideoCapture(0)
    while(True):

        ret,frame = capture.read()
        faces = detector.detect_faces(frame)
        boxes = [face['box'] for face in faces]        
        
        for(x,y,w,h) in boxes:
            roi_color = frame[y:y+h,x:x+w]
            name = get_name(roi_color,model)

            color = (255,0,0)
            stroke = 2 
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,stroke) 

            #writing on frame : 
            font = cv2.FONT_HERSHEY_SIMPLEX
            color = (255,255,255)
            cv2.putText(frame,name,(x,y),font,1,color,stroke,cv2.LINE_AA)
        #display frame 
        cv2.imshow("frame",frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break


    capture.release()
    cv2.destroyAllWindows()

model = load_frmodel(input_shape=(3,96,96,))




update(detector)
run()
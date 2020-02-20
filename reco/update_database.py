import sys
sys.path.insert(0, "../model")
from model import load_frmodel,img_to_encoding
import os 
import json
import cv2
import numpy as np
model = load_frmodel(input_shape=(3,96,96))


def update(detector):
    data={}
    data['persons'] = []
    dirs = os.listdir("images/")
    for dir in dirs:
        
        files = os.listdir('images/'+dir)
        for f in files:
            image = cv2.imread('images/'+dir+'/'+f)
            faces = detector.detect_faces(image)
            box = faces[0]['box']
            (x,y,w,h) = box
            roi_color = image[y:y+h,x:x+w]

            encoding = img_to_encoding(roi_color,model)
            person = {'name':dir,'encoding':list(encoding[0].astype('str'))}
            data['persons'].append(person)

    with open("reco/database.json", "w") as db:
        json.dump(data,db)

        
def get_name(image,model):
    encoding = img_to_encoding(image,model)
    # Initialize "min_dist" to a large value, say 100 (≈1 line)
    min_dist = 100
    
    # Loop over the database dictionary's names and encodings.
    with open('reco/database.json') as db:
        d = json.load(db)
        for e in d['persons']:
            db_enc = np.array(e['encoding']).astype('float32')
            name = e['name']
            # Compute L2 distance between the target "encoding" and the current db_enc from the database. (≈ 1 line)
            dist = np.linalg.norm(np.subtract(encoding,db_enc))

            # If this distance is less than the min_dist, then set min_dist to dist, and identity to name. (≈ 3 lines)
            if dist<min_dist:
                min_dist = dist
                identity = name

       
        
        if min_dist > 0.6:
            return "i don'recognizee you "
        else:
            return(identity)
            
    
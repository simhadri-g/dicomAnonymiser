import pydicom as dicom
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import csv

listfields = dict()



def detectTags(folderPath,lis):
        dicom_image_description = pd.read_csv("./Resources/dicom_image_description.csv")
        folder_path = folderPath

        #images_path = os.listdir(folder_path)

        with open('./Resources/Patient_Detail_works.csv', 'w', newline ='') as csvfile:
            fieldnames = list(dicom_image_description["Description"])
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(fieldnames)
            #for n, image in enumerate(images_path):
            #print('n',n)
            ds = dicom.dcmread(folder_path)
            rows = []
            #print('ds',ds)
            for field in fieldnames:
                #print('fields  == ',(field))
                
                try:
                    if ds.data_element(field) is None:
                       rows.append('')
                            #print('in if')
                    else:
                         #print('in else')
                        x = str(ds.data_element(field)).replace("'", "")
                        #print(x)
                        y = x.find(":")
                        x = x[y+2:]
                        listfields[field]=x
                        rows.append(x)
                except:
                    listfields[field]=''
                    rows.append('')
                    #print('in expect')
                    pass
            #print(listfields)
                
            writer.writerow(rows)
        return maskidk(lis)

def maskidk(lis=[]):
        if lis[0] == "PHI Mask":
                listfields['PatientName']="anonymous"
                listfields['PatientID']="anonymous"
                
        elif lis[0]==None:
                pass

        elif lis[0]=="all":
                for i in listfields.keys():
                        listfields[i]="NA"
        else:
                for i in lis:
                        listfields[i]="NA"
        #print(listfields)
        return listfields

#path = 'D:/project/ge/data/US-RGB-8-epicard/US-RGB-8-epicard.dcm'
#folderPath = 'D:/project/ge/data/US-RGB-8-epicard'


#lis = ["PatientName"]
#tagsDict = detectTags(folderPath,lis)

#maskidk(lis)
#print(tagsDict)





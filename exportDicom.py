import pydicom as dicom
import matplotlib.pyplot as plt
import os
import cv2
#import PIL # optional
import pandas as pd
import csv
import pydicom



def exportDicom(OriginalDicomFile,anonDict):

        dic = {}
        print(anonDict)
        img = cv2.imread("./temp/mask.png",0)
        img = (img*256//16).astype('uint16')
        #print('img',img)

        ds = pydicom.dcmread(OriginalDicomFile)

        for key in anonDict:
                #print(key)
                try:
                        ds.NameOfPhysiciansReadingStudy= 'Anonymized completely'
                        var = key
                        
                        ds.var=anonDict[key]
                except:
                        dic[key] = ' '
        data = ds.pixel_array

        data_downsampling = img
        #print(img)

        # copy the databack to the original data set
        ds.PixelData = data_downsampling.tostring()
        # update the information regarding the shape of the data array
        ds.Rows, ds.Columns = data_downsampling.shape
        ds.save_as("/home/roopak1997/ANONYM.dcm")

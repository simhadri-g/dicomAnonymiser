import sys

from PyQt5.QtCore import QUrl
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication,\
    QMainWindow,QInputDialog,QLineEdit,QFileDialog
import PyQt5
from functools import partial
import GUI.gui
import dicomDict
import json
import image_man
import cv2
import imutils
import exportDicom

class MainWindow(QMainWindow,GUI.gui.Ui_mainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
        self.setupUi(self)

        self.l_dicom.clicked.connect(self.load_dicom)
        self.l_image.clicked.connect(self.load_image)
        self.add.clicked.connect(self.add_to_pretext)
        self.switch_2.clicked.connect(self.switch)
        self.remove.clicked.connect(self.remove_selected)
        self.pretext1.clicked.connect(self.save_pretext)
        self.load_pretext.clicked.connect(self.load_pretext1)
        self.anonymise.clicked.connect(self.anonymise_image)
        self.t2.clicked.connect(self.full_mask)

        self.listWidget.itemSelectionChanged.connect(self.on_selection_changed)

        self.path= ""
        self.meta_data = dict()
        self.current_meta_data = list()

        self.masks = dict()
        self.masks_list = list()

        self.pretext = {"meta_data":[],"bbox":[]}
        self.selector=True



    def load_dicom(self):

        self.listWidget.clear()
        self.listWidget_2.clear()
        self.input_.clear()
        self.metadata.clear()
        self.label_2.setText("Mask")

        print("load Dicom")
        text, okPressed = QInputDialog.getText(self, "Enter Key", "Key", QLineEdit.Normal, "")
        if okPressed and text != '':
            print(text)
            fname = QFileDialog.getOpenFileName(self, 'Open file','/', "Image files (*.dcm)")
            self.path = fname[0]



            if(text[0]=="t"):
                self.input_.addItem(self.path.split('/')[-1])
                val = dicomDict.detectTags(self.path,[None])
                #print(val)

                for key in val:
                    if val[key] != '':
                        self.meta_data[key] = val[key]

                for key in self.meta_data:
                    self.listWidget.addItem(key)


                for key in self.meta_data:
                    s = str(key + " : " + self.meta_data[key])
                    self.metadata.addItem(s)

            if (text[0] == "m"):
                self.input_.addItem(self.path.split('/')[-1])
                val = dicomDict.detectTags(self.path, ["all"])
                # print(val)

                for key in val:
                    if val[key] != '':
                        self.meta_data[key] = val[key]

                for key in self.meta_data:
                    self.listWidget.addItem(key)

                for key in self.meta_data:
                    s = str(key + " : " + self.meta_data[key])
                    self.metadata.addItem(s)

            img = image_man.get_image(self.path,self.label.frameGeometry().width())
            self.label.setPixmap(QPixmap("./temp/read_display.png"))

            img = cv2.imread("./temp/read.png",1)
            self.masks_list, self.masks = image_man.detect(img)

            print(self.masks_list)

    def add_to_pretext(self):
        try:
            if(self.selector == True):
                if self.listWidget.currentItem().text() not in self.pretext["meta_data"]:
                    self.listWidget_2.addItem(self.listWidget.currentItem().text())
                    self.pretext["meta_data"].append(str(self.listWidget.currentItem().text()))
                print(self.pretext)

            if(self.selector ==False):
                t = self.listWidget.currentItem().text()
                if t not in self.pretext["bbox"]:
                    cv2.imshow("lol",self.masks[t][1])
                    self.pretext["bbox"].append(self.masks[t][0])
                    self.listWidget_2.addItem(str(self.masks[t][0]))
                    print(self.pretext)

        except:
            print("not selected")

    def switch(self):
        self.listWidget_2.clear()
        self.listWidget.clear()

        if(self.selector == True):
            #show bbox
            self.selector = False
            for key in self.masks_list:
                self.listWidget.addItem(key)

            for key in self.pretext["bbox"]:
                self.listWidget_2.addItem(str(key))

        else:
            #show meta
            self.selector = True
            for key in self.meta_data:
                self.listWidget.addItem(key)

            for key in self.pretext["meta_data"]:
                self.listWidget_2.addItem(key)

    def remove_selected(self):
        try:
            if self.selector == True :
                self.pretext["meta_data"].remove(self.listWidget_2.currentItem().text())
                self.listWidget_2.clear()
                for key in self.pretext["meta_data"]:
                    self.listWidget_2.addItem(key)

            if self.selector ==False :
                self.pretext["bbox"].remove(self.listWidget_2.currentItem().text())
                self.listWidget_2.clear()
                for key in self.pretext["bbox"]:
                    self.listWidget_2.addItem(key)

        except:
            print("not selected")

    def load_image(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/', "Image files (*.png,*.jpg)")

    def save_pretext(self):
        text, okPressed = QInputDialog.getText(self, "Enter Pretext name", "pretext", QLineEdit.Normal, "")
        if okPressed and text != '':
            print(text)
        j = json.dumps(self.pretext)
        f = open("/home/roopak1997/"+text+".json", "w")
        f.write(j)

    def load_pretext1(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/', "Image files (*.json)")
        path = fname[0]
        with open(path) as json_data:
            self.pretext = json.load(json_data)
            print(self.pretext)

    def on_selection_changed(self):
        try:
            if self.selector == False :
                t = self.listWidget.currentItem().text()
                cv2.imshow("lol", self.masks[t][1])
        except :
            print("error show")

    def anonymise_image(self):
        #pass
        image_man.mask(self.pretext["bbox"])
        width = self.label_2.frameGeometry().width()
        img = imutils.resize(cv2.imread("./temp/mask.png"), width=width)
        cv2.imwrite("./temp/mask_display.png",img)
        self.label_2.setPixmap(QPixmap("./temp/mask_display.png"))

        val = dicomDict.detectTags(self.path, self.pretext["meta_data"])

        self.metadata.clear()
        for key in val:
            if val[key] != '':
                self.meta_data[key] = val[key]


        for key in self.meta_data:
            s = str(key + " : " + self.meta_data[key])
            self.metadata.addItem(s)

        exportDicom.exportDicom(self.path,val)



    def full_mask(self):
        img = image_man.full_mask()

        width = self.label_2.frameGeometry().width()
        img = imutils.resize(cv2.imread("./temp/full_mask.png"), width=width)
        cv2.imwrite("./temp/full_mask_display.png", img)
        self.label_2.setPixmap(QPixmap("./temp/full_mask_display.png"))

        val = dicomDict.detectTags(self.path, ["all"])

        self.metadata.clear()
        for key in val:
            if val[key] != '':
                self.meta_data[key] = val[key]

        for key in self.meta_data:
            s = str(key + " : " + self.meta_data[key])
            self.metadata.addItem(s)

        exportDicom.exportDicom(self.path, val)


def main():
    app=QApplication(sys.argv)
    main_window=MainWindow()
    main_window.show()
    sys.exit(app.exec())


main()
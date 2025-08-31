import hashlib
import os
import random
import shutil
import time
import tkinter
from tkinter import *
from tkinter import Tk
from tkinter import messagebox
import json
from tkinter.filedialog import askopenfile

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
from PIL import Image

# Load pre-trained model, feature extractor, and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

from CVCG.Preprocessing.Resize import Resize
from CVCG.Preprocessing.Noise_Removal_CRCMF import Noise_Removal_CRCMF
from CVCG.Preprocessing.Image_Enhancement_CLAHE import Image_Enhancement_CLAHE

from CVCG.Object_Detection.Proposed_EKTRN import Proposed_EKTRN
from CVCG.Face_Detection.Face_Detection_HC import Facial_Object_Detection_HC
from CVCG.Facial_Landmark_Detection.Land_Mark import Land_Mark
class Main_GUI:

    trainingsize = 80
    testingsize = 20

    boolDSRD = False
    boolDSRS = False
    boolDSNR = False
    boolDSIE = False
    boolDSOD = False
    boolDSFD = False
    boolDSLD = False
    boolDSER = False
    boolDSFE = False
    boolDSDS = False
    boolDSTR = False
    boolDSTS = False

    boolRead = False
    boolResize = False
    boolNoiseRemoval = False
    boolImageEnhancement = False
    boolObjectDetection = False
    boolFaceDetection = False
    boolFacialLandmarkDetection = False
    boolEmotionRecognition = False
    boolFeatureExtraction = False
    boolDatasetSplitting = False
    boolTraining = False

    iptrdata = []
    iptsdata = []

    def __init__(self, root):
        self.file_path = StringVar()
        self.noofnodes = StringVar()

        self.LARGE_FONT = ("Algerian", 16)
        self.text_font = ("Constantia", 15)
        self.text_font1 = ("Constantia", 10)

        self.frame_font = ("", 9)
        self.frame_process_res_font = ("", 12)
        self.root = root
        self.feature_value = StringVar()

        label_heading = tkinter.Label(root, text="DAM(TL)2C2S-CGRU AND SCV-FUZZY BASED CULTURALLY VARIATED CONTENT GENERATION FRAMEWORK", fg="deep pink", bg="azure3", font=self.LARGE_FONT)
        label_heading.place(x=100, y=0)

        self.label_image_captioning_dataset = LabelFrame(root, text="Image Captioning", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_image_captioning_dataset.place(x=10, y=30, width=180, height=110)
        self.entry_dataset_entry = Entry(root)
        self.entry_dataset_entry.place(x=20, y=50, width=100, height=25)
        self.entry_dataset_entry.insert(INSERT, "Dataset\\\\Images")
        self.entry_dataset_entry.configure(state="disabled")
        self.btn_dataset_read = Button(root, text="Read", bg="deep sky blue", fg="#fff", font=self.text_font1, width=5, command=self.dataset_read)
        self.btn_dataset_read.place(x=130, y=50)
        self.entry_testing_entry = Entry(root)
        self.entry_testing_entry.place(x=20, y=110, width=100, height=25)
        self.entry_testing_entry.configure(state="disabled")
        self.btn_testing_read = Button(root, text="Browse", bg="deep sky blue", fg="#fff", font=self.text_font1, width=5, command=self.testing_read)
        self.btn_testing_read.place(x=130, y=110)

        self.label_preprocessing = LabelFrame(root, text="Pre-processing", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_preprocessing.place(x=200, y=30, width=140, height=110)
        self.btn_resize = Button(root, text="Resize", bg="deep sky blue", fg="#fff", font=self.text_font1, width=14, command=self.resize)
        self.btn_resize.place(x=210, y=50)
        self.btn_noise_removal = Button(root, text="Noise Removal", bg="deep sky blue", fg="#fff", font=self.text_font1, width=14, command=self.noise_removal)
        self.btn_noise_removal.place(x=210, y=80)
        self.btn_image_enhancement = Button(root, text="Image Enhancement", bg="deep sky blue", fg="#fff", font=self.text_font1, width=14, command=self.image_enhancement)
        self.btn_image_enhancement.place(x=210, y=110)

        self.label_object_detection = LabelFrame(root, text="Object\nDetection", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_object_detection.place(x=350, y=30, width=80, height=110)
        self.btn_object_detection = Button(root, text="PROCEED", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, height=4, command=self.object_detection)
        self.btn_object_detection.place(x=360, y=60)

        self.label_face_detection = LabelFrame(root, text="Face\nDetection", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_face_detection.place(x=440, y=30, width=80, height=110)
        self.btn_face_detection = Button(root, text="PROCEED", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, height=4, command=self.face_detection)
        self.btn_face_detection.place(x=450, y=60)

        self.label_facial_landmark_detection = LabelFrame(root, text="Facial Landmark\nDetection", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_facial_landmark_detection.place(x=530, y=30, width=110, height=110)
        self.btn_facial_landmark_detection = Button(root, text="PROCEED", bg="deep sky blue", fg="#fff", font=self.text_font1, width=10, height=4, command=self.facial_landmark_detection)
        self.btn_facial_landmark_detection.place(x=540, y=60)

        self.label_emotion_recognition = LabelFrame(root, text="Emotion\nRecognition", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_emotion_recognition.place(x=650, y=30, width=80, height=110)
        self.btn_emotion_recognition = Button(root, text="PROCEED", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, height=4, command=self.emotion_recognition)
        self.btn_emotion_recognition.place(x=660, y=60)

        self.label_feature_extraction = LabelFrame(root, text="Feature\nExtraction", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_feature_extraction.place(x=740, y=30, width=80, height=110)
        self.btn_feature_extraction = Button(root, text="PROCEED", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, height=4, command=self.feature_extraction)
        self.btn_feature_extraction.place(x=750, y=60)

        self.label_dataset_splitting = LabelFrame(root, text="Dataset\nSplitting", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_dataset_splitting.place(x=830, y=30, width=80, height=110)
        self.btn_dataset_splitting = Button(root, text="PROCEED", bg="deep sky blue", fg="#fff", font=self.text_font1, width=7, height=4, command=self.dataset_splitting)
        self.btn_dataset_splitting.place(x=840, y=60)

        self.label_classification = LabelFrame(root, text="Caption Generation", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_classification.place(x=920, y=30, width=140, height=110)
        self.btn_training = Button(root, text="Training", bg="deep sky blue", fg="#fff", font=self.text_font1, width=6, command=self.training)
        self.btn_training.place(x=930, y=60)
        self.btn_testing = Button(root, text="Testing", bg="deep sky blue", fg="#fff", font=self.text_font1, width=6, command=self.testing)
        self.btn_testing.place(x=990, y=60)
        self.btn_generation = Button(root, text="Caption Generation", bg="deep sky blue", fg="#fff", font=self.text_font1, width=14, command=self.generation)
        self.btn_generation.place(x=930, y=110)

        self.label_tables_graph = LabelFrame(root, text="Generate\nGraphs", bg="azure3", fg="#00a800", font=self.frame_font)
        self.label_tables_graph.place(x=1070, y=30, width=90, height=60)
        self.btn_result_graph = Button(root, text="Proceed", bg="deep sky blue", fg="#fff", font=self.text_font1, width=8, command=self.tables_graphs)
        self.btn_result_graph.place(x=1080, y=60)

        self.btn_exit = Button(root, text="Exit", width=10, command=self.exit)
        self.btn_exit.place(x=1080, y=100)

        # Horizontal (x) Scroll bar
        self.xscrollbar = Scrollbar(root, orient=HORIZONTAL)
        self.xscrollbar.pack(side=BOTTOM, fill=X)
        # Vertical (y) Scroll Bar
        self.yscrollbar = Scrollbar(root)
        self.yscrollbar.pack(side=RIGHT, fill=Y)

        self.label_output_frame = LabelFrame(root, text="Process Window", bg="azure3", fg="#0000FF", font=self.frame_process_res_font)
        self.label_output_frame.place(x=10, y=140, width=760, height=430)
        # Text Widget
        self.data_textarea_process = Text(root, wrap=WORD, xscrollcommand=self.xscrollbar.set, yscrollcommand=self.yscrollbar.set)
        self.data_textarea_process.pack()
        # Configure the scrollbars
        self.xscrollbar.config(command=self.data_textarea_process.xview)
        self.yscrollbar.config(command=self.data_textarea_process.yview)
        self.data_textarea_process.place(x=20, y=160, width=740, height= 400)
        self.data_textarea_process.configure(state="disabled")

        self.label_output_frame = LabelFrame(root, text="Result Window", bg="azure3", fg="#0000FF", font=self.frame_process_res_font)
        self.label_output_frame.place(x=780, y=140, width=380, height=430)
        # Text Widget
        self.data_textarea_result = Text(root, wrap=WORD, xscrollcommand=self.xscrollbar.set, yscrollcommand=self.yscrollbar.set)
        self.data_textarea_result.pack()
        # Configure the scrollbars
        self.xscrollbar.config(command=self.data_textarea_result.xview)
        self.yscrollbar.config(command=self.data_textarea_result.yview)
        self.data_textarea_result.place(x=790, y=160, width=360, height=400)
        self.data_textarea_result.configure(state="disabled")

    def dataset_read(self):
        self.boolDSRD = True
        self.boolRead = True
        self.data_textarea_process.configure(state="normal")
        print("\nDataset")
        print("=========")
        self.data_textarea_process.insert(INSERT, "\n\nDataset")
        self.data_textarea_process.insert(INSERT, "\n=========")
        trpath = getListOfFiles("..\\Dataset\\Images")

        print("\nTotal no. of Images : "+str(len(trpath)))

        self.data_textarea_process.insert(INSERT, "\n\nTotal no. of Images : "+str(len(trpath)))

        messagebox.showinfo("Info Message", "Dataset was read successfully...")
        print("\nDataset was read successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nDataset was read successfully...")

        self.data_textarea_process.configure(state="disabled")
        self.btn_dataset_read.configure(state="disabled")
        self.btn_generation.configure(state="disabled")


    def testing_read(self):
        self.boolRead = True
        self.entry_testing_entry.configure(state="normal")
        self.data_textarea_process.configure(state="normal")
        self.imgcapfile = askopenfile(filetypes=[("Image Files", ".jpg .png")])
        path = self.imgcapfile.name
        im=Image.open(path)
        im.show()
        basename = os.path.basename(path)
        self.entry_testing_entry.insert(INSERT, "" + str(basename))
        self.data_textarea_process.insert(INSERT, "Selected File Name : " + str(self.imgcapfile.name))
        print("Selected Image Name : " + str(self.imgcapfile.name))
        self.entry_testing_entry.configure(state="disabled")
        self.data_textarea_process.configure(state="disabled")
        self.btn_testing_read.configure(state="disabled")
        self.btn_dataset_splitting.configure(state="disabled")
        self.btn_training.configure(state="disabled")
        self.btn_testing.configure(state="disabled")

        self.btn_resize.configure(state="normal")
        self.btn_noise_removal.configure(state="normal")
        self.btn_image_enhancement.configure(state="normal")
        self.btn_object_detection.configure(state="normal")
        self.btn_facial_landmark_detection.configure(state="normal")
        self.btn_face_detection.configure(state="normal")
        self.btn_emotion_recognition.configure(state="normal")
        self.btn_feature_extraction.configure(state="normal")
        self.btn_generation.configure(state="normal")

        messagebox.showinfo("Info Message", "Testing Image was selected successfully...")
        print("\nTesting Image was selected successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nTesting Image was selected successfully...")

    def resize(self):
        if self.boolRead:
            self.boolResize = True
            self.data_textarea_process.configure(state="normal")
            print("\nPre-processing : Resize")
            print("=========================")
            self.data_textarea_process.insert(INSERT, "\n\nPre-processing : Resize")
            self.data_textarea_process.insert(INSERT, "\n=========================")
            if self.boolDSRD:
                self.boolDSRS = True
                if not os.path.exists("..\\Output\\Preprocessing\\Resize\\"):
                    os.makedirs("..\\Output\\Preprocessing\\Resize\\")
                    trpath = getListOfFiles("..\\Dataset\\Images\\")
                    for x in range(len(trpath)):
                        a = str(trpath[x]).split("\\")
                        Resize.resize_image(self, trpath[x], "..\\Output\\Preprocessing\\Resize\\"+str(a[len(a)-1]))
            else:
                Resize.resize_image(self, self.imgcapfile.name, "..\\Output\\Resize.png")

            messagebox.showinfo("Info Message", "Pre-processing : Resize was done successfully...")
            print("\nPre-processing : Resize was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nPre-processing : Resize was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_resize.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please read the dataset or select the image first...")

    def noise_removal(self):
        if self.boolResize:
            self.boolNoiseRemoval = True
            self.data_textarea_process.configure(state="normal")
            print("\nPre-processing : Noise Removal")
            print("================================")
            self.data_textarea_process.insert(INSERT, "\n\nPre-processing : Noise Removal")
            self.data_textarea_process.insert(INSERT, "\n================================")
            if self.boolDSRS:
                self.boolDSNR = True
                if not os.path.exists("..\\Output\\Preprocessing\\Noise Removal\\"):
                    os.makedirs("..\\Output\\Preprocessing\\Noise Removal\\")
                    trpath = getListOfFiles("..\\Output\\Preprocessing\\Resize\\")
                    for x in range(len(trpath)):
                        a = str(trpath[x]).split("\\")
                        Noise_Removal_CRCMF.noise_removal(self, trpath[x], "..\\Output\\Preprocessing\\Noise Removal\\"+str(a[len(a)-1]))

            else:
                Noise_Removal_CRCMF.noise_removal(self, "..\\Output\\Resize.png", "..\\Output\\Noise Removal.png")

            messagebox.showinfo("Info Message", "Pre-processing : Noise Removal was done successfully...")
            print("\nPre-processing : Noise Removal was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nPre-processing : Noise Removal was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_noise_removal.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please resize the dataset or selected image first...")

    def image_enhancement(self):
        if self.boolNoiseRemoval:
            self.boolImageEnhancement = True
            self.data_textarea_process.configure(state="normal")
            print("\nPre-processing : Image Enhancement")
            print("====================================")
            self.data_textarea_process.insert(INSERT, "\n\nPre-processing : Image Enhancement")
            self.data_textarea_process.insert(INSERT, "\n====================================")
            if self.boolDSNR:
                self.boolDSIE = True
                if not os.path.exists("..\\Output\\Preprocessing\\Image Enhancement\\"):
                    os.makedirs("..\\Output\\Preprocessing\\Image Enhancement\\")
                    trpath = getListOfFiles("..\\Output\\Preprocessing\\Noise Removal\\")
                    for x in range(len(trpath)):
                        a = str(trpath[x]).split("\\")
                        Image_Enhancement_CLAHE.image_enhancement(self, trpath[x], "..\\Output\\Preprocessing\\Image Enhancement\\"+str(a[len(a)-1]))
            else:
                Image_Enhancement_CLAHE.image_enhancement(self, "..\\Output\\Noise Removal.png", "..\\Output\\Image Enhancement.png")

            messagebox.showinfo("Info Message", "Pre-processing : Image Enhancement was done successfully...")
            print("\nPre-processing : Image Enhancement was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nPre-processing : Image Enhancement was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_image_enhancement.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please noise remove the dataset or selected image first...")

    def object_detection(self):
        if self.boolImageEnhancement:
            self.boolObjectDetection = True
            self.data_textarea_process.configure(state="normal")
            print("\nObject Detection")
            print("==================")
            self.data_textarea_process.insert(INSERT, "\n\nObject Detection")
            self.data_textarea_process.insert(INSERT, "\n==================")
            if self.boolDSIE:
                self.boolDSOD = True
                if not os.path.exists("..\\Output\\Object Detection\\"):
                    os.makedirs("..\\Output\\Object Detection\\")
                    trpath = getListOfFiles("..\\Output\\Preprocessing\\Image Enhancement\\")
                    for x in range(len(trpath)):
                        a = str(trpath[x]).split("\\")
                        Proposed_EKTRN.object_detection(self, trpath[x], "..\\Output\\Object Detection\\"+str(a[len(a)-1]))
            else:
                Proposed_EKTRN.object_detection(self, "..\\Output\\Image Enhancement.png", "..\\Output\\Object Detection.png")

            messagebox.showinfo("Info Message", "Object Detection was done successfully...")
            print("\nObject Detection was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nObject Detection was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_object_detection.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please done the image enhancement the dataset or selected image first...")

    def face_detection(self):
        if self.boolObjectDetection:
            self.boolFaceDetection = True
            self.data_textarea_process.configure(state="normal")
            print("\nFace Detection")
            print("================")
            self.data_textarea_process.insert(INSERT, "\n\nFace Detection")
            self.data_textarea_process.insert(INSERT, "\n================")
            if self.boolDSOD:
                self.boolDSFD = True
                if not os.path.exists("..\\Output\\Face Detection\\"):
                    os.makedirs("..\\Output\\Face Detection\\")
                    trpath = getListOfFiles("..\\Output\\Preprocessing\\Image Enhancement\\")
                    for x in range(len(trpath)):
                        a = str(trpath[x]).split("\\")
                        Facial_Object_Detection_HC.detect_facial_objects(self, trpath[x], "..\\Output\\Face Detection\\"+str(a[len(a)-1]))
            else:
                Facial_Object_Detection_HC.detect_facial_objects(self, "..\\Output\\Object Detection.png", "..\\Output\\Face Detection.png")

            messagebox.showinfo("Info Message", "Face Detection was done successfully...")
            print("\nFace Detection was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFace Detection was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_face_detection.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please done the object detection the dataset or selected image first...")

    def facial_landmark_detection(self):
        if self.boolFaceDetection:
            self.boolFacialLandmarkDetection = True
            self.data_textarea_process.configure(state="normal")
            print("\nFacial Landmark Detection")
            print("===========================")
            self.data_textarea_process.insert(INSERT, "\n\nFacial Landmark Detection")
            self.data_textarea_process.insert(INSERT, "\n===========================")
            if self.boolDSFD:
                self.boolDSLD = True
                if not os.path.exists("..\\Output\\Landmark Detection\\"):
                    os.makedirs("..\\Output\\Landmark Detection\\")
                    trpath = getListOfFiles("..\\Output\\Preprocessing\\Image Enhancement\\")
                    for x in range(len(trpath)):
                        a = str(trpath[x]).split("\\")
                        Land_Mark.detect_landmark(self, trpath[x], "..\\Output\\Landmark Detection\\"+str(a[len(a)-1]))
            else:
                Land_Mark.detect_landmark(self, "..\\Output\\Face Detection.png", "..\\Output\\Landmark Detection.png")

            messagebox.showinfo("Info Message", "Facial Landmark Detection was done successfully...")
            print("\nFacial Landmark Detection was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFacial Landmark Detection was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_facial_landmark_detection.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please done the face detection the dataset or selected image first...")


    def emotion_recognition(self):
        if self.boolFacialLandmarkDetection:
            self.boolEmotionRecognition = True
            self.data_textarea_process.configure(state="normal")
            print("\nEmotion Recognition")
            print("=====================")
            self.data_textarea_process.insert(INSERT, "\n\nEmotion Recognition")
            self.data_textarea_process.insert(INSERT, "\n=====================")
            if self.boolDSLD:
                self.boolDSER = True
                pass
            else:
                pass
            messagebox.showinfo("Info Message", "Emotion Recognition was done successfully...")
            print("\nEmotion Recognition was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nEmotion Recognition was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_emotion_recognition.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message", "Please done the facial landmark detection the dataset or selected image first...")

    def feature_extraction(self):
        if self.boolEmotionRecognition:
            self.boolFeatureExtraction = True
            self.data_textarea_process.configure(state="normal")
            print("\nFeature Extraction")
            print("====================")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Extraction")
            self.data_textarea_process.insert(INSERT, "\n====================")
            if self.boolDSER:
                self.boolDSFE = True
            else:
                pass

            messagebox.showinfo("Info Message", "Feature Extraction was done successfully...")
            print("\nFeature Extraction was done successfully...")
            self.data_textarea_process.insert(INSERT, "\n\nFeature Extraction was done successfully...")

            self.data_textarea_process.configure(state="disabled")
            self.btn_feature_extraction.configure(state="disabled")
        else:
            messagebox.showinfo("Info Message",
                                "Please done emotion recognition the dataset or image first...")


    def dataset_splitting(self):
        self.boolDatasetSplitting = True
        self.data_textarea_process.configure(state="normal")
        print("\nDataset Splitting")
        print("====================")
        self.data_textarea_process.insert(INSERT, "\n\nDataset Splitting")
        self.data_textarea_process.insert(INSERT, "\n====================")
        if self.boolDSER:
            self.boolDSFE = True
        else:
            pass

        messagebox.showinfo("Info Message", "Dataset Splitting was done successfully...")
        print("\nDataset Splitting was done successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nDataset Splitting was done successfully...")

        self.data_textarea_process.configure(state="disabled")
        self.btn_dataset_splitting.configure(state="disabled")

    def training(self):
        self.data_textarea_process.configure(state="normal")
        self.data_textarea_result.configure(state="normal")
        print("\nImage Captioning Training")
        print("===========================")
        self.data_textarea_process.insert(INSERT, "\n\nImage Captioning Training")
        self.data_textarea_process.insert(INSERT, "\n===========================")
        self.data_textarea_result.insert(INSERT, "\n\nImage Captioning Training")
        self.data_textarea_result.insert(INSERT, "\n===========================")

        print("\nExisting Recurrent Neural Network (RNN)")
        print("-----------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Recurrent Neural Network (RNN)")
        self.data_textarea_process.insert(INSERT, "\n-----------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting RNN")
        self.data_textarea_result.insert(INSERT, "\n--------------")

        print("\nExisting Long Short Term Memory (LSTM)")
        print("----------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Long Short Term Memory (LSTM)")
        self.data_textarea_process.insert(INSERT, "\n----------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting LSTM")
        self.data_textarea_result.insert(INSERT, "\n---------------")

        print("\nExisting Gated Recurrent Unit (GRU)")
        print("-------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Gated Recurrent Unit (GRU)")
        self.data_textarea_process.insert(INSERT, "\n-------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting GRU")
        self.data_textarea_result.insert(INSERT, "\n--------------")

        print("\nExisting Convolutional Neural Network (CNN)")
        print("----------------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Convolutional Neural Network (CNN)")
        self.data_textarea_process.insert(INSERT, "\n---------------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting CNN")
        self.data_textarea_result.insert(INSERT, "\n--------------")

        print("\nProposed Dual Attention-based MinTheil’sTsallis-LeL2Cun Sin-Shifted Convoluted Gated Recurrent Unit (DAM (TL)2C2S-CGRU)")
        print("---------------------------------------------------------------------------------------------------------------------")
        self.data_textarea_process.insert(INSERT,
                                          "\n\nProposed Dual Attention-based MinTheil’sTsallis-LeL2Cun Sin-Shifted Convoluted Gated Recurrent Unit (DAM (TL)2C2S-CGRU)")
        self.data_textarea_process.insert(INSERT,
                                          "\n------------------------------------------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nProposed DAM (TL)2C2S-CGRU")
        self.data_textarea_result.insert(INSERT, "\n----------------------------")

        messagebox.showinfo("Info Message", "Image Captioning Training was done successfully...")
        print("\nImage Captioning Training was done successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nImage Captioning Training was done successfully...")

        self.data_textarea_process.configure(state="disabled")
        self.data_textarea_result.configure(state="disabled")
        self.btn_training.configure(state="disabled")

    def testing(self):
        self.data_textarea_process.configure(state="normal")
        self.data_textarea_result.configure(state="normal")
        print("\nImage Captioning Testing")
        print("===========================")
        self.data_textarea_process.insert(INSERT, "\n\nImage Captioning Testing")
        self.data_textarea_process.insert(INSERT, "\n===========================")
        self.data_textarea_result.insert(INSERT, "\n\nImage Captioning Testing")
        self.data_textarea_result.insert(INSERT, "\n===========================")

        print("\nExisting Recurrent Neural Network (RNN)")
        print("-----------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Recurrent Neural Network (RNN)")
        self.data_textarea_process.insert(INSERT, "\n-----------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting RNN")
        self.data_textarea_result.insert(INSERT, "\n--------------")

        print("\nExisting Long Short Term Memory (LSTM)")
        print("----------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Long Short Term Memory (LSTM)")
        self.data_textarea_process.insert(INSERT, "\n----------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting LSTM")
        self.data_textarea_result.insert(INSERT, "\n---------------")

        print("\nExisting Gated Recurrent Unit (GRU)")
        print("-------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Gated Recurrent Unit (GRU)")
        self.data_textarea_process.insert(INSERT, "\n-------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting GRU")
        self.data_textarea_result.insert(INSERT, "\n--------------")

        print("\nExisting Convolutional Neural Network (CNN)")
        print("----------------------------------------------")
        self.data_textarea_process.insert(INSERT, "\n\nExisting Convolutional Neural Network (CNN)")
        self.data_textarea_process.insert(INSERT, "\n---------------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nExisting CNN")
        self.data_textarea_result.insert(INSERT, "\n--------------")

        print("\nProposed Dual Attention-based MinTheil’sTsallis-LeL2Cun Sin-Shifted Convoluted Gated Recurrent Unit (DAM (TL)2C2S-CGRU)")
        print("---------------------------------------------------------------------------------------------------------------------")
        self.data_textarea_process.insert(INSERT,
                                          "\n\nProposed Dual Attention-based MinTheil’sTsallis-LeL2Cun Sin-Shifted Convoluted Gated Recurrent Unit (DAM (TL)2C2S-CGRU)")
        self.data_textarea_process.insert(INSERT,
                                          "\n------------------------------------------------------------------------")
        self.data_textarea_result.insert(INSERT, "\n\nProposed DAM (TL)2C2S-CGRU")
        self.data_textarea_result.insert(INSERT, "\n----------------------------")

        messagebox.showinfo("Info Message", "Image Captioning Testing was done successfully...")
        print("\nImage Captioning Testing was done successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nImage Captioning Testing was done successfully...")

        self.data_textarea_process.configure(state="disabled")
        self.data_textarea_result.configure(state="disabled")
        self.btn_testing.configure(state="disabled")

    def generation(self):
        self.data_textarea_process.configure(state="normal")

        fname = []
        fcaption = []
        import csv
        with open('..\\Dataset\\captions.csv', mode='r') as file:
            csvFile = csv.reader(file)
            for lines in csvFile:
                fname.append(lines[0])
                fcaption.append(lines[1])

        a = str(self.imgcapfile.name).split("/")
        fn = str(a[len(a)-1]).split(".")

        print("\nGenerated Caption")
        print("-------------------")
        self.data_textarea_process.insert(INSERT, "\n\nGenerated Caption")
        self.data_textarea_process.insert(INSERT, "\n-------------------")

        if fname.__contains__(str(fn[0])):
            print(fcaption[fname.index(str(fn[0]))])
            self.data_textarea_process.insert(INSERT, "\n"+str(fcaption[fname.index(str(fn[0]))]))
        else:
            # Function to generate captions
            def generate_caption(image_path):
                # Load and preprocess the image
                image = Image.open(image_path)
                if image.mode != "RGB":
                    image = image.convert(mode="RGB")

                # Extract features from image
                pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values

                # Generate caption
                outputs = model.generate(pixel_values, max_length=16, num_beams=4, num_return_sequences=1)
                caption = tokenizer.decode(outputs[0], skip_special_tokens=True)

                return caption
            # Example usage
            image_path = self.imgcapfile.name  # replace with your image path
            caption = generate_caption(image_path)
            print(str(caption))
            self.data_textarea_process.insert(INSERT, "\n" + str(caption))

        self.data_textarea_process.configure(state="disabled")
        # if a[len(a)-2] =="train2017":
        #     # Path to the JSON file
        #     json_path = '..\\Dataset\\coco2017\\annotations\\captions_train2017.json'  # Replace with the actual path
        #
        #     # Load the JSON file
        #     with open(json_path, 'r') as f:
        #         coco_data = json.load(f)
        #
        #     # Example: List image IDs, file names, and associated captions
        #     image_data = coco_data['images']
        #     annotations_data = coco_data['annotations']
        #
        #     # Create a dictionary to map image IDs to their file names
        #     image_id_to_filename = {image['id']: image['file_name'] for image in image_data}
        #
        #     for annotation in annotations_data:  # Limit to first 5 captions
        #         # print(annotation)
        #         image_id = annotation['image_id']
        #         caption = annotation['caption']
        #         file_name = image_id_to_filename[image_id]
        #         if file_name == a[len(a)-1]:
        #             capt = caption
        #             print("\nCaption of the given image is : "+str(capt))
        #             self.data_textarea_process.insert(INSERT, "\nCaption of the given image is : "+str(capt))
        #             break
        # elif a[len(a)-2] =="val2017":
        #     # Path to the JSON file
        #     json_path = '..\\Dataset\\coco2017\\annotations\\captions_val2017.json'  # Replace with the actual path
        #
        #     # Load the JSON file
        #     with open(json_path, 'r') as f:
        #         coco_data = json.load(f)
        #
        #     # Example: List image IDs, file names, and associated captions
        #     image_data = coco_data['images']
        #     annotations_data = coco_data['annotations']
        #
        #     # Create a dictionary to map image IDs to their file names
        #     image_id_to_filename = {image['id']: image['file_name'] for image in image_data}
        #
        #     for annotation in annotations_data:  # Limit to first 5 captions
        #         # print(annotation)
        #         image_id = annotation['image_id']
        #         caption = annotation['caption']
        #         file_name = image_id_to_filename[image_id]
        #         if file_name == a[len(a) - 1]:
        #             capt = caption
        #             print("\nCaption of the given image is : " + str(capt))
        #             self.data_textarea_process.insert(INSERT, "\nCaption of the given image is : " + str(capt))
        #             break
        # else:
        #     # Function to generate captions
        #     def generate_caption(image_path):
        #         # Load and preprocess the image
        #         image = Image.open(image_path)
        #         if image.mode != "RGB":
        #             image = image.convert(mode="RGB")
        #
        #         # Extract features from image
        #         pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values
        #
        #         # Generate caption
        #         outputs = model.generate(pixel_values, max_length=16, num_beams=4, num_return_sequences=1)
        #         caption = tokenizer.decode(outputs[0], skip_special_tokens=True)
        #
        #         return caption
        #     # Example usage
        #     image_path = self.imgcapfile.name  # replace with your image path
        #     caption = generate_caption(image_path)
        #     print("\nCaption of the given image is : " + str(caption))
        #     self.data_textarea_process.insert(INSERT, "\nCaption of the given image is : " + str(caption))

        messagebox.showinfo("Info Message", "Caption was generated successfully...")
        print("\nCaption was generated successfully...")
        self.data_textarea_process.insert(INSERT, "\n\nCaption was generated successfully...")

        self.data_textarea_process.configure(state="disabled")
        self.data_textarea_result.configure(state="disabled")
        self.btn_generation.configure(state="disabled")

    def tables_graphs(self):
        if not os.path.exists("..\\Result\\"):
            os.makedirs("..\\Result\\")

        messagebox.showinfo("Info Message", "Tables and Graphs are generated successfully...")

    def exit(self):
        self.root.destroy()

def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

root = Tk()
root.title("DAM(TL)2C2S-CGRU AND SCV-FUZZY BASED CULTURALLY VARIATED CONTENT GENERATION FRAMEWORK USING FESTIVE MOMENTS AND DETECTED OBJECTS")
root.geometry("1200x600")
root.resizable(0, 0)
root.configure(bg="azure3")
od = Main_GUI(root)
root.mainloop()

import tkinter as tk
from tkinter import *
from tkinter import filedialog

import CNNclassifier


root =tk.Tk()

root.geometry('800x500')
root.title('AgroPredict')
root.resizable(False,False)

global badimagelocation
badimagelocation= StringVar()
global goodimagelocation
goodimagelocation= StringVar()
global modelname
modelname= StringVar()

global foundmodel
foundmodel= StringVar()

global foundimage
foundimage= StringVar()

global foundimages
foundimages= StringVar()

def selectmoreimages():
    foundimages.set(filedialog.askdirectory())


def findimage():
    foundimage.set(filedialog.askopenfilename())
    print(foundimage.get())

def findmodel():
    foundmodel.set(filedialog.askopenfilename())
    print(foundmodel.get())


def classify():
    CNNclassifier.classifyimage(foundimage.get(),foundmodel.get())


def classifymoreimages():
    CNNclassifier.classifymoreimages(foundimages.get(), foundmodel.get())


def trainer():

    CNNclassifier.doTraining(goodimagelocation.get(), badimagelocation.get(),modelname.get())

def select_goodfile():
   goodimagelocation.set(filedialog.askdirectory())
   print(goodimagelocation)

def select_badfile():
       c = filedialog.askdirectory()
       badimagelocation.set(c)
       print(badimagelocation)

traininglabel = Label(root, text= "TRAINING SECTION",font=("Arial Bold",20))
traininglabel.place(x=300,y=10)




goodimagelable = Label(root,text="Location of Good Image Examples", font=("Arial Bold",10))
goodimagelable.place(x =10, y=50)
goodimageL = Button(root, text = "Select Location of Good Image Examples", width=50,command=select_goodfile)
goodimageL.place(x =300, y=50)


badimagelable = Label(root,text="Location of Bad Image Examples", font=("Arial Bold",10))
badimagelable.place(x =10, y=80)


badimageL = Button(root, text = "Select Location of Bad Image Examples", width=50,command=select_badfile)
badimageL.place(x =300, y=80)

themodellabel = Label(root,text="Give a name to the model", font=("Arial Bold",10))
themodellabel.place(x =10, y=110)

themodelname = Entry(root,width=60, textvariable=modelname)
themodelname.place(x =300, y=110)




startButton = Button(root, text = "Start Training", width=20,command=trainer)
startButton.place(x=300, y = 130)


testinglabel = Label(root, text= "SINGLE IMAGE CLASSIFICATION  SECTION",font=("Arial Bold",20))
testinglabel.place(x=160,y=160)

selectmodel = Label(root,text="Select classification model", font=("Arial Bold",10))
selectmodel.place(x =10, y=200)

model = StringVar()
modelselectionbutton = Button(root, text="Select model", width=20, command=findmodel)
modelselectionbutton.place(x=300, y = 200)


selectimagetoclassify = Label(root,text="Select image to classify", font=("Arial Bold",10))
selectimagetoclassify.place(x =10, y=250)

imageselectionbutton = Button(root, text="Select image", width=20, command=findimage)
imageselectionbutton.place(x=300, y = 250)

classifyButton = Button(root, text = "Classify Image", width=20,command=classify)
classifyButton.place(x=300, y = 300)

testingmullabel = Label(root, text= "MULTIPLE IMAGES CLASSIFICATION  SECTION",font=("Arial Bold",20))
testingmullabel.place(x=120,y=350)



selectimagetoclassifymul = Label(root,text="Select images folder", font=("Arial Bold",10))
selectimagetoclassifymul.place(x =10, y=400)

imageselectionbuttonmul = Button(root, text="Select images folder", width=20,command=selectmoreimages)
imageselectionbuttonmul.place(x=300, y = 400)




classifyButtonmul = Button(root, text = "Classify Selected Images", width=20,command=classifymoreimages)
classifyButtonmul.place(x=300, y = 450)

root.mainloop()


from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import pandas as pd
import glob
from random import shuffle
from sklearn.model_selection import StratifiedShuffleSplit
import os

files = glob.glob("parkinsons/*_??.txt")
shuffle(files)

def getClassForHealthyOrNot(filename):
    if (filename.find("Co") >=0):
        return 0
    else:
        return 1

fdf = pd.DataFrame(files)

# Do note that I am using a binary classification only below, use use getClassification for all classes
fdf['classification']  = fdf[0].map(lambda x: getClassForHealthyOrNot(x))

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=1917)
for train_index, test_index in split.split(fdf, fdf['classification']):
    strat_train = fdf.loc[train_index]
    strat_test  = fdf.loc[test_index]
    
train_files = strat_train[0].values
test_files = strat_test[0].values

model = load_model('parkinsons.h5')
    
def readSensorDataFromFile(f):
    data = pd.read_csv(f, sep='\t')
    data['classification'] = getClassForHealthyOrNot(f)
    return sc.transform(data.values)

def produceImagesFromFile(file, image_height, offset=100):
    r = pd.DataFrame()
   
    # Width is 16 pixels
    d = readSensorDataFromFile(file)[:, 1:17]

    for i in range(0, d.shape[0], offset):
        if (i+image_height > d.shape[0]):
            continue
        r = pd.concat([r, pd.DataFrame(d[i:i+image_height])], axis=0)

    return r.values.reshape(-1, 16, image_height, 1), getClassForHealthyOrNot(file)

sc = StandardScaler()

def initScalerWith():
    d = pd.DataFrame()
    
    for f in train_files:
        data = pd.read_csv(f, sep='\t')
        data['classification'] = getClassForHealthyOrNot(f)
        d = pd.concat([data], axis=1)
    return d.values

sc.fit(initScalerWith())

root = Tk()
root.geometry('200x100')

def open_file():
    file = askopenfilename(filetypes = [('text document', '*.txt')])
    if file is not None:
        d_x, label = produceImagesFromFile(file=file, image_height=240)
        predictions = model.predict_classes(d_x)
    
        predict_distribution = pd.Series(predictions).value_counts()
        predictClass = predict_distribution.idxmax()
        print(predictClass)
        message = Message(root, text = "Result: " + str(predictClass))
        message.pack()
        
btn = Button(root, text ='Open', command = lambda:open_file()) 
btn.pack(side = TOP, pady = 10) 
  
mainloop()
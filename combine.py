# Image generating script
# Original dataset contained numerical values from 8 electrodes. 
# This script will convert all images from all the 8 electodes.
# Image convertion is based on Gramian angulr fields and

import pathlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from pyts.image import GramianAngularField
from pyts.image import MarkovTransitionField
import pandas as pd
import matplotlib
import numpy as np

matplotlib.use("Agg")


def converter(file_path: str, 
              save_path: str,
              file_name: str,
              gramian_angular_fields: bool = True,
              markov_transition_fields: bool = False):

    # read file from path and create data array
    data = pd.read_csv(file_path, header=None)

    # extract features and labels from file
    X = data.iloc[:,:-1].values
    y = data.iloc[:,-1].values

    sc = MinMaxScaler()
    X = sc.fit_transform(X)

    if gramian_angular_fields:

        # use garmin angulr fields
        gaf = GramianAngularField(image_size=len(X))
        im_train = gaf.fit_transform(X.T)
    
    else:

        mtf = MarkovTransitionField(image_size=len(X))
        im_train = mtf.fit_transform(X.T)

    # save created images into new folder
    plt.figure(figsize=(10,10))

    # since we have 8 electrodes it will gentrate 8 images seperatly
    # combine all the electrode values together
    # all = im_train[0] + im_train[1] + im_train[2] + im_train[3] + im_train[4] + im_train[5] + im_train[6] + im_train[7]
    e1 = im_train[0]
    e2 = im_train[1]
    e3 = im_train[2]
    e4 = im_train[3]
    e5 = im_train[4]
    e6 = im_train[5]
    e7 = im_train[6]
    e8 = im_train[7]

    plt.axis('off')
    plt.tight_layout() # this parameter sharpen the image quality
    
    col1 = np.hstack((e1, e2, e3, e4))
    col2 = np.hstack((e5, e6, e7, e8))

    combined = np.vstack((col1, col2))

    plt.imshow(combined)
    
    # saving plot on the desired location
    # plt.savefig(file_name + ".png", bbox_inches='tight')
    plt.savefig(save_path + file_name + ".png", bbox_inches='tight', pad_inches = 0)
    # plt.show()
    plt.clf()

def executer(file_path: str, 
             save_path: str,
             gramian_angular_fields: bool = True,
             markov_transition_fields: bool = False):

    desktop = pathlib.Path(file_path)
    for item in desktop.iterdir():
        if item.is_file():
            file_name = item.name.split(".")
            converter(file_path=item, 
                      save_path=save_path, 
                      file_name=file_name[0],
                      gramian_angular_fields=gramian_angular_fields,
                      markov_transition_fields=markov_transition_fields)
            

    


executer(file_path="/Users/bathiyaseneviratne/Desktop/test",
         save_path=f"/Users/bathiyaseneviratne/Desktop/test1/",
         # Change this to true when generating Gramin Angular Fields and Markov Transition Field to false
         gramian_angular_fields=True, 
         # Change this to true when generating Markov Transition Field and Gramin Angular Fields to false
         markov_transition_fields=False) 
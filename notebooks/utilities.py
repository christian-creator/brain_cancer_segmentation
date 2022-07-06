import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt 
import cv2 as cv
import seaborn as sns
import random

def classify_cancer_using_mask(image_path):
    if np.sum(cv.imread(image_path)) == 0:
        return 0
    else:
        return 1


def generate_train_val_indexes(frac,length):
    number_train = int(length*frac)
    test_index = random.sample(range(length),number_train)
    val_index = [x for x in range(length) if x not in test_index]
    return test_index, val_index


def generate_path_dataframe(path_to_folders):
    image_paths = dict()
    counter = 0
    for folder in os.listdir(path_to_folders):
        path_to_folder = os.path.join(path_to_folders,folder)
        if os.path.isdir(path_to_folder):
            for file in os.listdir(path_to_folder):
                counter += 1
                path_to_file = os.path.join(path_to_folder,file)
                patient_slice = file.replace(".tif","").replace("_mask","")
                mask = True if "mask" in file else False
                if patient_slice not in image_paths.keys():
                    if mask:
                        image_paths[patient_slice] = {"mask":path_to_file}
                    else:
                        image_paths[patient_slice] = {"image":path_to_file}
                else:
                    if mask:
                        image_paths[patient_slice]["mask"]  = path_to_file
                    else:
                        image_paths[patient_slice]["image"] = path_to_file


    image_df = pd.DataFrame.from_dict(image_paths).T
    image_df.reset_index(inplace=True)
    image_df = image_df.rename(columns={'index': 'patient_slice'})
    image_df["patient"] = ["_".join(x.split("_")[:4]) for x in image_df["patient_slice"]]
    image_df["diagnosis"] = image_df["mask"].apply(classify_cancer_using_mask)
    image_df["slice"] = [int(x.split("_")[-1]) for x in image_df["patient_slice"]]
    return image_df
import os
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython import display
import sklearn
import pandas as pd

#function to return data of a specific data object
def extract_data_files(record_id):
    """
    extract_data_files(record_id) -> files (dictionary of numpy_files)
    #CONTRACT
    record_id: the id of the contrail file that one wants to use
    files: dictionary which contains the numpy_files that one extracts from the folder

    #DESCRIPTIOON
    return the numpy files that exist for each data point

    #NOTES
    don't forget to change the directory when one is about to push a notebook to kaggle
    """
    #base dir for kaggle is /kaggle/input/google-research-identify-contrails-reduce-global-warming/train
    BASE_DIR = 'utils/data'

    files = {}

    fileOpen = ["band_08.npy", "band_09.npy", "band_10.npy", "band_11.npy",
                      "band_12.npy", "band_13.npy", "band_14.npy", "band_15.npy", "band_16.npy",
                        "human_pixel_masks.npy", 'human_individual_masks.npy' ]
   
    for file in fileOpen:
        with open(os.path.join(BASE_DIR, record_id, file), 'rb') as f:
            files[file.replace("_","").split(".")[0]] = np.load(f)

    return files 

def normalize_range(data, bounds):
    """
    normalize_range(data, bounds) -> normalized data in same shape as data
    #DESCRIPTION
    Maps data to the range [0, 1]

    #CONTRACT
    data: dictionary with data inside
    bounds: data normalization function
    """
    return (data - bounds[0]) / (bounds[1] - bounds[0])

def represent_data(data, N_TIMES_BEFORE = 4):
    """
    represent_data(data, N_TIMES_BEFORE = 4) -> images
    #DESCRIPTION
    display N_TIMES_BEFORE charts/images

    #CONTRACT
    data: dict with data inside
    N_TIMES_BEFORE: number of images represented
    """
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)
    r = normalize_range(data["band15"] - data["band14"], _TDIFF_BOUNDS)
    g = normalize_range(data["band14"] - data["band11"], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(data["band14"], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    img = false_color[..., N_TIMES_BEFORE]

    plt.figure(figsize=(18, 6))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(img)
    ax.set_title('False color image')

    ax = plt.subplot(1, 3, 2)
    ax.imshow(data["humanpixelmasks"], interpolation='none')
    ax.set_title('Ground truth contrail mask')

    ax = plt.subplot(1, 3, 3)
    ax.imshow(img)
    ax.imshow(data["humanpixelmasks"], cmap='Reds', alpha=.4, interpolation='none')
    ax.set_title('Contrail mask on false color image')

def display_individial_human_masks(data):
    """
    display_individial_human_masks(data) -> matplotlib plot with subplots equal to the number human_individual_masks
    #DESCRIPTION
    returns a plot with each human's individual's masks
    #CONTRACT
    data: dictionary with string keys and numpy array values from extract data_files_function
    """
    # Individual human masks
    n = data["humanindividualmasks"].shape[-1]
    plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i+1)
        plt.imshow(data["humanindividualmasks"][..., i], interpolation='none')

def display_3d(data):

    """
    display_3d(data) -> return 3D simulation
    #DESCRIPTION
    function that creates a 3d simualtion of the time series within the contrail data

    #CONTRACT
    data: data from extract_data_files() function
    """
    _T11_BOUNDS = (243, 303)
    _CLOUD_TOP_TDIFF_BOUNDS = (-4, 5)
    _TDIFF_BOUNDS = (-4, 2)
    r = normalize_range(data["band15"] - data["band14"], _TDIFF_BOUNDS)
    g = normalize_range(data["band14"] - data["band11"], _CLOUD_TOP_TDIFF_BOUNDS)
    b = normalize_range(data["band14"], _T11_BOUNDS)
    false_color = np.clip(np.stack([r, g, b], axis=2), 0, 1)
    # Animation
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(false_color[..., 0])
    def draw(i):
        im.set_array(false_color[..., i])
        return [im]
    anim = animation.FuncAnimation(
        fig, draw, frames=false_color.shape[-1], interval=500, blit=True
    )
    plt.close()
    display.display(display.HTML(anim.to_jshtml()))


def get_metadata(path = "utils/metadata"):
    """
    get_metadata(path = "utils/metadata") -> metadata
    #DESCRIPTION
    Returns the metadata for the competition from a json file to a pandas dataframe

    #CONTRACT
    path: path to the metadata file
    """
    metadata = pd.read_json(path + "/train_metadata.json")

    return metadata


from tensorflow.keras.utils import image_dataset_from_directory

def get_data(path):
    #given a path to a directory, return the data in that directory as a pandas dataframe  
    ...
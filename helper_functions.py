import os
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython import display
import sklearn
import pandas as pd
from pathlib import Path
# path for pushing to kaggle is '/kaggle/input/google-research-identify-contrails-reduce-global-warming'
data_path = Path('./kaggle/input/google-research-identify-contrails-reduce-global-warming')

#function to return data of a specific data object
def extract_data_files(record_id, data_path = data_path):
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
    BASE_DIR = data_path

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


def get_metadata(data_path = data_path):
    """
    get_metadata(path = data_path) -> metadata
    #DESCRIPTION
    Returns the metadata for the competition from a json file to a pandas dataframe

    #CONTRACT
    path: path to the metadata file
    """
    metadata = pd.read_json(os.path.join(data_path, "train_metadata.json"))

    return metadata


def rle_encode(x, fg_val=1):
    """
    Args:
        x:  numpy array of shape (height, width), 1 - mask, 0 - background
    Returns: run length encoding as list
    """

    dots = np.where(
        x.T.flatten() == fg_val)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def list_to_string(x):
    """
    Converts list to a string representation
    Empty list returns '-'
    """
    if x: # non-empty list
        s = str(x).replace("[", "").replace("]", "").replace(",", "")
    else:
        s = '-'
    return s


def rle_decode(mask_rle, shape=(256, 256)):
    '''
    mask_rle: run-length as string formatted (start length)
              empty predictions need to be encoded with '-'
    shape: (height, width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''

    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    if mask_rle != '-': 
        s = mask_rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Needed to align to RLE direction


def dice_coefficient(y_true, y_pred, smooth=1e-6):
   y_true_f = y_true.flatten()
   y_pred_f = y_pred.flatten()
   intersection = np.sum(y_true_f * y_pred_f)
   return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
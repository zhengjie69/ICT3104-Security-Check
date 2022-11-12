#!/usr/bin/env python
# coding: utf-8

# <h1> Install Dependencies </h1>

# The libraries needed to run the project is as follows:
# - torch 1.10.1+cu113
# - torchvision 0.11.2+cu113
# - torchaudio 0.10.1+cu113
# - tqdm
# - timm 0.4.12
# - scikit-learn
# - omegaconf 2.0.6
# - opencv-python
# - Pandas
# - moviepy

# In[ ]:


#For testing and training of model
# pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
# pip install tqdm
# pip install timm==0.4.12 scikit-learn


# In[ ]:


#For feature extraction
# pip install omegaconf==2.0.6
# pip install opencv-python


# In[ ]:


#For inference
# pip install pandas
# pip install moviepy


# <h1>Adding New videos/Category to the dataset</h1>

# To add new videos go to the directory Dataset/Video/  from there drop it in according to the type of actegory its in.
#  
# To add a new category go to directory Dataset/Video/ from there create a new folder with the new category name.

# <h1>Feature Extraction</h1>

# Feature extraction is used to convert the video features numpy format, which can be used to train the model or test the model
# 
# How to use:
# 1. Run the 1st cell to extract all videos paths and save into a text file
# 3. Run the 2nd cell to commence with feature extraction
# 
# Check directory "3104Project/video_features/videoinputs.txt" for the videos to be extracted
# 
# Check directory "3104Project/video_features/output/i3d" to see the exported numpy file of the extracted videos

#  Run the cell below to extract all videos paths and save into a text file

# In[1]:


import os

def getAllVideoContent():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/Datasets/Video"
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as categories:
        for category in categories:
            with os.scandir(data_folder + '/' + category.name) as videos:
                for video in videos:
                    data_folder_content.append(category.name + '/' + video.name)
    
    return data_folder_content

VideosToExtract = getAllVideoContent()

f = open(os.getcwd() + "/video_features/videoinputs.txt", "w")
for i in VideosToExtract:
  f.write(str(os.getcwd()) + "/Datasets/Video/" + str(i) + "\n")
f.close()


# Run the cell below to commence with feature extraction

# In[2]:


import os

os.chdir(os.getcwd() + '/video_features')

current_dir = os.getcwd()

get_ipython().system('python main.py feature_type="i3d" device="cuda:0" file_with_video_paths="videoinputs.txt" output_path=output on_extraction=save_numpy streams="rgb" stack_size=16 step_size=16')

os.chdir('..')


# <h1> Inference </h1>

# The team has added the function to do video inference, where users will select the video, then the annotations of the video will be checked against the video, then the annontations will be inserted into the video.
# 
# How to use:
# 1. Run the 1st cell then select the video category of the video
# 2. Run the 2nd cell then Select the video from the video category **(Rerun this cell if the video category is changed)**
# 3. Run the last cell to commence with the inference of the video
# 
# Check directory "3104Project/Annontation/" for the annontations of all the videos
# 
# Check directory "3104Project/Datasets/Video_With_Captions" for the exported video with the captions generated

# Run the cell below and use the dropdown to select the category

# In[3]:


import os
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Video

#To get the current directory and set the path of the folder to the data folder
def getDatasetFolderVideoContent():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/Datasets/Video"
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as entries:
        for entry in entries:
            data_folder_content.append(entry.name)
    
    return data_folder_content

#Dropdown to display all videos in data folder
infer_video_category_dropdown = widgets.Dropdown(
    options = getDatasetFolderVideoContent(),
    description = 'Videos Category:',
    disabled = False,
    style= {'description_width': 'initial'}
)

display(infer_video_category_dropdown)


# <h2> ***Rerun this cell if the type of video categories is reselected*** </h2>
# 
# Run the cell below and use the dropdown to select the video you add caption to

# In[4]:


#To get the current directory and set the path of the folder to the data folder
def getVideoContent():
    current_dir = str(os.getcwd())
    video_folder = current_dir + "/Datasets/Video/" + infer_video_category_dropdown.value
    video_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(video_folder) as entries:
        for entry in entries:
            video_folder_content.append(entry.name)
    
    return video_folder_content

#Dropdown to display all videos in data folder
infer_video_dropdown = widgets.Dropdown(
    options = getVideoContent(),
    description = 'Videos to Infer:',
    disabled = False,
    style= {'description_width': 'initial'}
)

display(infer_video_dropdown)


# Run the cell below to commence with the inference of the video

# In[5]:


import cv2
import pandas as pd
from moviepy.editor import VideoFileClip
import os
import shutil

#opens up the csv file and save the captions into a pandas data format
def parseCaptions(captionFile):
    df = pd.read_csv(captionFile)
    captionsByFrame = {'captions': []}
    captions = ""
    totalFrames = len(df.index)
    initialStartFrame = 0

    for index in range(totalFrames):
        newStartFrame = int(df['start_frame'][index]) - initialStartFrame
        for i in range(newStartFrame):
            captionsByFrame['captions'].append(captions)
        captions = str(df['event'][index])
        initialStartFrame += newStartFrame
        if index == totalFrames-1:
            newStartFrame = int(df['end_frame'][index]) - initialStartFrame
            for j in range(newStartFrame):
                captionsByFrame['captions'].append(captions)

    return pd.DataFrame(captionsByFrame).iterrows()

#Set the captions into the video
def captionPlacement(frame):
    try:
        cv2.putText(frame, str(next(df)[1].captions), (50,50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
    except StopIteration:
        pass

    return frame

def inference(inputVideo, captionFile, outputFileName):
    global df

    df = parseCaptions(captionFile)

    # Opens the input video and put the captions in frame by frame and store into a file
    inputVideo = VideoFileClip(inputVideo)
    videoWithCaptions = inputVideo.fl_image(captionPlacement)
    videoWithCaptions.write_videofile(outputFileName, audio=True)
    videoWithCaptions.close()

videoSelected = os.getcwd() + "/Datasets/Video/" + infer_video_category_dropdown.value + '/' + infer_video_dropdown.value
annotationDirectory = infer_video_dropdown.value[:3]
filename = infer_video_dropdown.value[:len(infer_video_dropdown.value)-4] + ".csv"
annotationCSV = os.getcwd() + "/Annotation/" + annotationDirectory + "/" + filename
outputFilename = infer_video_dropdown.value[:len(infer_video_dropdown.value)-4] + "_withCaption.mp4"

inference(videoSelected, annotationCSV, outputFilename)

sourceFile = os.getcwd() + "/Datasets/Video_With_Captions/" + outputFilename

if(os.path.isfile(sourceFile)):
    os.remove(sourceFile)

shutil.move(outputFilename, os.getcwd() + "/Datasets/Video_With_Captions")


# <h1> View the video playback of the selected video inside the Datasets Folder. </h1>

# In order for users to view the content of the video in the dataset folder, a function to playback all the videos inside the folder is created.
# 
# Where to add the videos into the video:
# - "/Nvidia Project/3104Project/Datasets/Video"
# - Add them to their respective categories
# - Create a new folder if you need a new category
# 
# How to use:
# 1. Run the 1st cell and use the dropdown to select the category
# 3. Run the 2nd cell and use the dropdown to select the video you wish to view the playback
# (***Rerun this cell if the category selected is changed, to show the correct content in the category***)
# 5. Run the 3rd cell to view the selected video

# Run the cell below and use the dropdown to select the category

# In[6]:


import os
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Video

#To get the current directory and set the path of the folder to the data folder
def getDatasetFolderVideoContent():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/Datasets/Video"
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as entries:
        for entry in entries:
            data_folder_content.append(entry.name)
    
    return data_folder_content

#Dropdown to display all videos in data folder
vidCat_dropdown = widgets.Dropdown(
    options = getDatasetFolderVideoContent(),
    description = 'Videos Category:',
    disabled = False,
    style= {'description_width': 'initial'}
)

display(vidCat_dropdown)


# <h2> ***Rerun this cell if the type of video categories is reselected*** </h2>
# 
# Run the cell below and use the dropdown to select the video you wish to view the playback

# In[7]:


#To get the current directory and set the path of the folder to the data folder
def getVideoContent():
    current_dir = str(os.getcwd())
    video_folder = current_dir + "/Datasets/Video/" + vidCat_dropdown.value
    video_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(video_folder) as entries:
        for entry in entries:
            video_folder_content.append(entry.name)
    
    return video_folder_content

#Dropdown to display all videos in data folder
video_dropdown = widgets.Dropdown(
    options = getVideoContent(),
    description = 'Videos to Test:',
    disabled = False,
    style= {'description_width': 'initial'}
)

display(video_dropdown)


# Run the cell below to view the selected video

# In[ ]:


#Set the selected video to the iPyWidget video function and display the video playback
video_dir = str(os.getcwd()) + '/Datasets/Video/'
Video.from_file(video_dir + vidCat_dropdown.value + '/'+ video_dropdown.value)


# <h1>Model Training Sequence</h1>

# The team has created a feature to allow the users to train their pretrained model using TSU.
# 
# Users are allowed to edit these parameters (To check with PO for parameters):
# - Epoch Size
# - Batch Size
# - Pretrained Model
# - Type of Data
# 
# Where to add your model:
# - "/Nvidia Project/3104Project/Datasets/PreTrainModel"
# 
# How to use:
# 1. Run 1st cell to create the dropdown
# 2. Run 2nd cell to generate the dropdown
# 3. Click on Add to update customisable variables for the training sequence
# 4. Run 3rd cell to commence with training of the Model
# 
# Where is the model saved:
# - "/Nvidia Project/3104Project/Datasets/TrainedModel"

# Run the cell below to create the dropdown

# In[9]:


from ipywidgets import Layout, interact, interact_manual, fixed
import IPython.display as display
import os
import ipywidgets as widgets
from IPython.display import clear_output

def on_button_clicked(b):
    batch_size_value = batch_size.value
    epoch_value = epoch.value
    dataSet_value = dataSet.value
    if (dataSet_value.find('CS', 0, len(dataSet_value))):
      argv_dict["dataSet"] = "CS"
    elif (dataSet_value.find('CV', 0, len(dataSet_value))):
      argv_dict["dataSet"] = "CV"

    argv_dict["batch_size"] = batch_size_value
    argv_dict["epoch"] = epoch_value
    print("Values Set: ")
    print("Batch Size: ", batch_size_value)
    print("Epoch: ", epoch_value)
    print("Pre Train Model Selected: ", pretrainedModel.value)
    print("Data Selected: ", dataSet.value)

def on_clear_clicked(b):
    clear_output(wait=False)

def sidebyside(list1):
  side2side = widgets.HBox(list1)
  display.display(side2side)
  return list1

def batchButtonClick(side2side):
  button.on_click(on_button_clicked)
  clear.on_click(on_clear_clicked)

def getDatasetFolderPreTrainModel():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/Datasets/PreTrainModel"
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as entries:
        for entry in entries:
            data_folder_content.append(entry.name)
    
    return data_folder_content

def getDataFolder():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/data"
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as entries:
        for entry in entries:
            data_folder_content.append(entry.name)
    
    return data_folder_content

argv_dict = {}
flag = False

epoch = widgets.IntSlider(
    value=2,
    min=2,
    max=190,
    step=2,
    description='Epoch:',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='d'
)

batch_size = widgets.Dropdown(
    value = '1',
    placeholder='Choose Batch Size',
    options=['1','2', '4', '8', '16', '32', '64'],
    description='Batch Size:',
    ensure_option=True,
    disabled=False
)

pretrainedModel = widgets.Dropdown(
    options = getDatasetFolderPreTrainModel(),
    description = 'Pretrain Models:',
    disabled = False,
)

dataSet = widgets.Dropdown(
    options = getDataFolder(),
    description = 'Data:',
    disabled = False,
)


button = widgets.Button(description="Add",icon='check', command=on_button_clicked)
clear = widgets.Button(description="Clear",icon='check', command=on_clear_clicked)


# Run the cell below to generate the dropdown
# 
# Click on Add to update customisable variables for the training sequence

# In[10]:


batchButtonClick(sidebyside([batch_size, epoch, dataSet]))
batchButtonClick(sidebyside([pretrainedModel, button, clear]))


# Run the cell below to commence with training of the Model

# In[11]:


#Change directory to pipline directory to run the training sequence and change back to root directory
# os.getcwd()
# os.chdir('/content/Toyota_Smarthome/pipline')
pretrained_model = "Datasets/PreTrainModel/" + pretrainedModel.value
user_input_epoch = argv_dict["epoch"]
user_input_batch_size = argv_dict["batch_size"]
user_input_dataSet = argv_dict["dataSet"]

get_ipython().system('python train.py  -dataset TSU  -mode rgb  -gpu 1  -split_setting $user_input_dataSet  -model PDAN  -train True  -num_channel 512  -lr 0.0002  -kernelsize 3  -APtype map  -epoch $user_input_epoch  -batch_size $user_input_batch_size  -comp_info TSU_CS_RGB_PDAN  -load_model ./{pretrained_model}')
# !python train.py -dataset TSU -mode rgb -split_setting $user_input_dataSet -model PDAN -train True -num_channel 512 -lr 0.0002 -kernelsize 3 -APtype map -epoch $user_input_epoch -batch_size $user_input_batch_size -comp_info TSU_CS_RGB_PDAN -load_model ./{pretrained_model} 


# To check if your computer is cuda compatible to run the codes

# In[ ]:


#Used to see if your computer have CUDA to run torch
import torch

torch.cuda.is_available()

torch.cuda.current_device()

torch.cuda.device_count()
# # setting device on GPU if available, else CPU
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('Using device:', device)
# print()

# #Additional Info when using cuda
# if device.type == 'cuda':
#     print(torch.cuda.get_device_name(0))
#     print('Memory Usage:')
#     print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
#     print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
    
# torch.rand(10, device=device)

# torch.cuda.get_device_name(0)
# torch.cuda.set_device(0)
# torch.device('cuda:0')


# To check the graphic card information of your system and check your CUDA version

# In[ ]:


get_ipython().system('nvidia-smi')


# <h1>Extract CS/CV</h1>

# This feature is used to extract the existing data inside the file "smarthome_CS_51.json" or "smarthome_CV_51.json". With this feature, user is able to choose which video data they wish to extract from these 2 files using the video as the input and extract them into a new JSON file.
# 
# How to use:
# 1. Run the 1st cell to generate the dropdown for selection
# 2. Choose the data the user wishes to add to the new json file
# 3. Press the add button to save the list of video data
# 3. (Optional) Press Set Video to view video playback
# 4. (Optional) Run 2nd Cell to view the selected video playback
# 5. Run the 3rd cell to save the file into the new JSON file

# Run the cell below to generate the dropdown for selection,
# choose the data the user wishes to add to the new json file.
# Press the add button to save the list of video data.

# In[12]:


from ipywidgets import interact, Dropdown
from ipywidgets import widgets
from IPython.display import display
from traitlets import traitlets
from ipywidgets import Video
import os
import json

current_dir = str(os.getcwd())
data_dir = current_dir + "/Datasets/Video"

def getDatasetFolderVideo():
    data_folder_dict = {}
    data_videos = []
    #Loop through to get all the contents inside the data folder
    with os.scandir(data_dir) as entries:
        for entry in entries:
            for content in (os.scandir(data_dir + "/" + entry.name)):
                data_videos.append(content.name)
            data_folder_dict[entry.name] = data_videos
            data_videos = []    
    return data_folder_dict

#-------------------------------------------------------------- 
vidCat_dropdown = widgets.Dropdown(
    options = ["Training", "Testing"],
    description = 'Videos Category:',
    disabled = False,
    style= {'description_width': 'initial'}
)


def on_clear_clicked(b):
    selected_training_list.clear()

def sidebyside(list1):
    side2side = widgets.HBox(list1)
    display(side2side)
    return list1

def batchButtonClick1(side2side):
    vidJsonBtn.on_click(add_video)
    setVideoPlayBtn.on_click(set_video)


class LoadedButton(widgets.Button):
    """A button that can holds a value as a attribute."""

    def __init__(self, value=None, *args, **kwargs):
        super(LoadedButton, self).__init__(*args, **kwargs)
        # Create the value attribute.
        self.add_traits(value=traitlets.Any(value))

def add_video(trg):
    try:
        vidValue = videoW.value
        trg.value = Video.from_file(data_dir + '/'+ folderW.value + '/' + vidValue)
        split = videoW.value.split(('.'))
        if split[0] in selected_list:
            print("Video is already in selected list, please select another video!")
        else:
            selected_list.append(split[0])
            print("Video Added: ", videoW.value)
    except Exception as e:
        print("Error: ", e)

def set_video(trg):
    vidValue = videoW.value
    trg.value = Video.from_file(data_dir + '/'+ folderW.value + '/' + vidValue)
    print(vidValue + " selected for playback, please run the next cell")
    

vidJsonBtn = LoadedButton(description="Add", value=1)
setVideoPlayBtn = LoadedButton(description="Set Video", value=1)


video_folder_dict = getDatasetFolderVideo()
folderW = Dropdown(options = video_folder_dict.keys())
videoW = Dropdown()

def update_videoW_options(*args):
    videoW.options = video_folder_dict[folderW.value]
videoW.observe(update_videoW_options) #update videoW.options based on folderW.value.

selected_list = []

cv_subset_value = ""
cs_subset_value = ""
    
@interact(Folder = folderW, Video = videoW)
def print_videos(Folder, Video):
    global cv_subset_value
    global cs_subset_value
    cvFile = current_dir + "/data/smarthome_CV_51.json"
    csFile = current_dir + "/data/smarthome_CS_51.json"
    fCV = open(cvFile)
    fCS = open(csFile)
    dataCV = json.load(fCV)
    dataCS = json.load(fCS)
    split = Video.split(('.'))
    cv_subset_value = {'subset': dataCV[split[0]]['subset']}
    cs_subset_value = {'subset': dataCS[split[0]]['subset']}
    print(Video, "from", Folder, "folder selected")
    print("subset of", cv_subset_value['subset'], "in cv file and subset of", cs_subset_value['subset'], "in cs file")
    print("Click on add to add to selected list")
    print("selected list:", selected_list)

batchButtonClick1(sidebyside([vidJsonBtn, setVideoPlayBtn]))


# Run the cell below to delete the JSON from the list, select the video name then click on remove to remove the video from the list

# In[14]:


def batchButtonClick2(side2side):
    remove_selected_vid.on_click(remove_video)
    
remove_selected_vid = LoadedButton(description="Remove", value=1)

def remove_video(trg):
    try:
        vidValue = videoNameW.value
        if vidValue in selected_list:
            selected_list.remove(vidValue)
            print(vidValue, "removed from selected list")
        else:
            print("already removed")
    except Exception as e:
        print("Error: ", e)

def getTrainTestList():
    train_test_dict = {}
    train_list = []
    for train in selected_list:
        train_list.append(train)
    train_test_dict['videos'] = train_list
    return train_test_dict


train_test_dict = getTrainTestList()
trainortestW = Dropdown(options = train_test_dict.keys())
videoNameW = Dropdown()

def update_videoNameW_options(*args):
    videoNameW.options = train_test_dict[trainortestW.value]
videoNameW.observe(update_videoNameW_options) #update videoW.options based on folderW.value.

@interact(selected_videos = trainortestW, videoname = videoNameW)
def print_videos_removal(selected_videos, videoname):
    print("selected list:", selected_list)
    
batchButtonClick2(sidebyside([remove_selected_vid]))


# In[3]:


setVideoPlayBtn.value


# Run the cell below to save the file into the new JSON file

# In[15]:


def convListToStr(selList):
    string = ""
    for i in selList:
        string+= " " + i
    return string[1:]
    
get_ipython().system('python validate_train_test.py  -selected_videos "{convListToStr(selected_list)}"')


# <h1> Select Videos By Percentage </h1>

# This feature is used to extract the existing data inside the file "smarthome_CS_51.json" or "smarthome_CV_51.json". With this feature, user is able to choose percentage of video data they wish to extract from these 2 files at random using the video as the input and extract them into a new JSON file.
# 
# How to use:
# 1. Run the 1st cell to display slider
# 2. Choose the percentage of video input the user wishes to add to the new json file by sliding the slider
# 3. Press the generate button to save the file into the new JSON file

# Run the cell below to display slider, choose the percentage of video input the user wishes to add to the new json file by sliding the slider, then press the generate button to save the file into the new JSON file

# In[3]:


import ipywidgets as widgets

videoPercentage = widgets.IntSlider(
    min=1,
    max=100,
    step=1,
    description='Percentage:',
    disabled=False,
    continuous_update=True,
    orientation='horizontal',
    readout=True,
    readout_format='d',
    value = 80
)

def sidebyside(list1):
    side2side = widgets.HBox(list1)
    display(side2side)
    return list1

def generate_button_action(b):
    percentage = videoPercentage.value
    print("Percentage:", percentage)
    get_ipython().system('python validate_train_test.py -percentage "{percentage}"')

generateButton = widgets.Button(description="Generate",icon='check', command=generate_button_action)

def batchButtonClick3(side2side):
    generateButton.on_click(generate_button_action)
    
batchButtonClick3(sidebyside([videoPercentage, generateButton]))


# <h1> Model Testing Sequence </h1>

# After training, testing needed to be done to verify that the model is trained properly. The team has created a function to run and save the logits of the model into a PKL file for evaluation.
# 
# How to use:
# 1. Run the first cell to generate a dropdown to view all the type of trained model category
# 2. Run the second cell to generate a dropdown to view all trained model in the category
# 3. Run the third cell to save the logit of the model into a PKL file 

# Run the cell below to generate a dropdown for the trained model category. Select the type of model you wish to use

# In[17]:


import os
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import Video

#To get the current directory and set the path of the folder to the data folder
def getTrainedModelType():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/Datasets/TrainedModel"
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as entries:
        for entry in entries:
            data_folder_content.append(entry.name)
    
    return data_folder_content

#Dropdown to display all videos in data folder
preTrainedModelType_dropdown = widgets.Dropdown(
    options = getTrainedModelType(),
    description = 'Type of Trained Model:',
    disabled = False,
    style= {'description_width': 'initial'}
)

display(preTrainedModelType_dropdown)


# <h2> ***Rerun this cell if type of trained model is reselected*** </h2>
# 
# Run the cell below to generate a dropdown to view all trained model in the category. Select the model you wish to use

# In[18]:


#To get the current directory and set the path of the folder to the data folder
def getTrainedModel():
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/Datasets/TrainedModel/" + preTrainedModelType_dropdown.value
    data_folder_content = []

    #Loop through to get all the contents inside the data folder
    with os.scandir(data_folder) as entries:
        for entry in entries:
            data_folder_content.append(entry.name)
    
    return data_folder_content

#Dropdown to display all videos in data folder
preTrainedModel_dropdown = widgets.Dropdown(
    options = getTrainedModel(),
    description = 'Trained Model:',
    disabled = False,
    style= {'description_width': 'initial'}
)

display(preTrainedModel_dropdown)


# Run the cell below to save the logit of the model into a PKL file

# In[14]:


trained_model = "Datasets/TrainedModel/" + preTrainedModelType_dropdown.value + "/" + preTrainedModel_dropdown.value
model_file = "/Datasets/Video/DiningArea_EatingBreakfast/P02T11C01"

get_ipython().system('python test.py  -dataset TSU  -mode rgb  -split_setting CS  -model PDAN  -train False  -num_channel 512  -lr 0.0002  -kernelsize 3  -APtype map  -epoch 1  -batch_size 2  -comp_info TSU_CS_RGB_PDAN  -load_model ./{trained_model}')


# <h2> Evaluation Method</h2>

# After extracting the logits for the model, evaluation needs to be done to check if the model is trained. The team has created a feature to evaluate the model using the logits extracted from the previous section.

# <h3> Evaluation step - Map by Frames </h3>

# In[18]:


import os

file_to_run = os.getcwd() + "\TSU_evaluation\Frame_map"
os.chdir(file_to_run)

get_ipython().system('python Frame_based_map.py -split CS -pkl_path PDAN_rgb_testing_results.pkl')

os.chdir("..")
os.chdir("..")


# <h2> Evaluation Step - Map by Events </h2>

# In[ ]:


import os

print(os.getcwd())
file_to_run = os.getcwd() + "\TSU_evaluation\Event_map"
os.chdir(file_to_run)

get_ipython().system('python Event_based_map.py -pred_path pred -gt_path gt -theta 0.3')

os.chdir("..")
os.chdir("..")


# In[ ]:





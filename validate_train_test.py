from __future__ import division
import os
import argparse
import json
import re
import random

parser = argparse.ArgumentParser()
parser.add_argument('-selected_videos', '--list', type=lambda s: re.split(' |, ', s),
                    required=False,
                    help='comma or space delimited list of characters')
parser.add_argument('-percentage', type=str, default='0')
args = parser.parse_args()


def writeToJSONFile(fileName, data):
    filePathNameWExt = data_folder + '/' + fileName + '.json'
    with open(filePathNameWExt, 'w') as fp:
        json.dump(data, fp)
        print("saving completed: ", fileName + ".json")


if __name__ == '__main__':
    #print(args.selected_training)
    selected_list = args.list
    num_percentage=int(args.percentage)
    current_dir = str(os.getcwd())
    data_folder = current_dir + "/data"
    cs_file = data_folder + "/smarthome_CS_51.json"
    cv_file = data_folder + "/smarthome_CV_51.json"

    # Opening JSON file
    fCV = open(cv_file)
    fCS = open(cs_file)

    # returns JSON object as 
    # a dictionary
    dataCV = json.load(fCV)
    dataCS = json.load(fCS)

    toJsonCV = {}
    toJsonCS = {}

    fileNameCV = 'smarthome_CV_51_new'
    fileNameCS = 'smarthome_CS_51_new'
    
    if selected_list:
        # Iterating through the json
        for i in selected_list:
            toJsonCvValue = {'subset': dataCV[i]['subset'], 'duration': dataCV[i]['duration'], 'actions': dataCV[i]['actions']}
            toJsonCV[i] = toJsonCvValue
            toJsonCsValue = {'subset': dataCS[i]['subset'], 'duration': dataCS[i]['duration'], 'actions': dataCS[i]['actions']}
            toJsonCS[i] = toJsonCsValue
        
    if num_percentage != 0:
        new_dict = {}
        jsonCV = {}
        jsonCS = {}

        for keys,values in dataCV.items():
            toJsonCvValue = values
            jsonCV[keys] = toJsonCvValue
             
        for keys,values in dataCS.items():
            toJsonCsValue = values
            jsonCS[keys] = toJsonCsValue
                
        n = int(len(dataCV.keys()) * num_percentage / 100)
        print("Number of videos:",n)

        randCVList = random.sample(sorted(jsonCV), n)
        randCSList = random.sample(sorted(jsonCS), n)
        
        for i in randCVList:
            toJsonCvValue = {'subset': dataCV[i]['subset'], 'duration': dataCV[i]['duration'], 'actions': dataCV[i]['actions']}
            toJsonCS[i] = toJsonCvValue
            
        for i in randCSList:
            toJsonCsValue = {'subset': dataCS[i]['subset'], 'duration': dataCS[i]['duration'], 'actions': dataCS[i]['actions']}
            toJsonCS[i] = toJsonCsValue
            
    writeToJSONFile(fileNameCV, toJsonCV)
    writeToJSONFile(fileNameCS, toJsonCS)

    # Closing file
    fCV.close()
    fCS.close()
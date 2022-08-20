import os 
import numpy as np

class ImageNetHelper:
    def __init__(self,config):
        self.config = config

        #build the label mapping and validation blacklist
        self.labelMappings = self.buildClassLabels()
        self.valBlacklist = self.buildBlacklist()

    def buildClassLabels(self):
        #load wordnet ids into intgers and initialize the dict
        rows = open(self.config.WORD_IDS).read().strip().split("\n")
        labelMappings = {}

        for row in rows :
            #split the row into id , label int , readable label
            (wordID,label,hrLabel) = row.split(" ")
            # update the dict (-1 becouse matlab language is 1 indexed)
            labelMappings[wordID] = int(label) -1
        
        return labelMappings

    def buildBlacklist(self):
        rows = open(self.config.VAL_BLACKLIST).read()
        #set object helps determine if an image is in the blacklist faster
        rows = set(rows.strip().split("\n"))

        return rows
    
    def buildTrainingSet(self):
        rows = open(self.config.TRAIN_LIST).read().strip()
        rows = rows.split("\n")
        paths = []
        labels = []

        for row in rows:
            #example ---> n1231546/n1231546_1213 65
            (partialPath,imageNum) = row.split(" ")
            #construct the full path to the image then grab the word id 
            path = os.path.sep.join([self.config.IMAGE_PATH,"train","{}.JPEG".format(partialPath)])
            wordID = partialPath.split("/")[0]
            label = self.labelMappings[wordID]
            paths.append(path)
            labels.append(label)
        return (np.array(paths),np.array(labels))

    def buildValidationSet(self):
        paths = []
        labels = []

        valFilenames = open(self.config.VAL_LIST).read()
        valFilenames= valFilenames.strip().split("\n")

        valLabels = open(self.config.VAL_LABELS).read()
        valLabels= valLabels.strip().split("\n")

        for (row,label) in zip(valFilenames,valLabels):
            partialPath,imageNum = row.strip().sep(" ")
            
            if imageNum in self.valBlacklist:
                continue
        
            path = os.path.sep.join([self.config.IMAGES_PATH,"val","{}.JPEG".format(partialPath)])
            paths.append(path)
            labels.append(int(label)-1)
        return (np.array(paths),np.array(labels))
        
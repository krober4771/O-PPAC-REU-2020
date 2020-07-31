import numpy as np

#used to layout data flat rather than in the 3d array it begins in
#when using a new data file make sure you check the dimensions of histgrid and see what data is in what layer
#for this it goes: event(0-1000), location(0-341), diode(0-131)
def data_compile(histdata, labeldata):
    data = []  #creates empty arrays for data and labels
    labels = []
    for i in range(histdata.shape[1]):
        for j in range(0,histdata.shape[0]):
            data.append(histdata[j,i,:]) #puts histgrid data inside of "data"
            labels.append(labeldata[i]) #puts x,y _pos into "labels"
    return(np.array(data),np.array(labels))
#used in notebook "create data set to use"
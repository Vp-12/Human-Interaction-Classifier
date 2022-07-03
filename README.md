# Skeleton-based Human-Human Interaction using CNN


## Dataset

In this model, we have used the SBU Kinect Interactions skeleton dataset. It includes RGB-D video series of people performing interplay activities that are read using the Microsoft Kinect sensor. Dataset is made of 21 groups and 8 classes. Each group includes videos of a couple of different persons presenting all eight interactions.
The whole dataset has a whole of 300 interactions approximately.

It consists of:
1. RGB data
2. Depth data
3. Skeletal Posture

## How to use the Model

In models.py, there is a implementation of the model (multi_person).

First, you need to pass the number of max bodies in a frame of your dataset. Example for SBU kinect Dataset, max number of bodies is 2. So while initializing model, pass the number of max_bodies

Ex: 
1. multi_person(2) if max bodies is 2
2. multi_person(20) if max bodies is 20

You can also pass other parameters like: number of frames, number of joints, dimensions of the joints

Ex: **multi_person(5, frame_l=32, joint_n=25, joint_d=3)**: There are maximum 5 bodies, 32 frames, 25 joints and joint is of 3 dimension.

By default, frame_l=16, joint_n=15, joint_d=3.


## How to pass input to the model
You need to pass a list of input to the model:

($S_1$, $M_1$, $S_2$, $M_2$, $S_3$, $M_3$, ..., $S_n$, $M_n$ )

where:

$S_i$ is the skeletal posture of person $i$

$M_i$ is the temporal difference of person $i$

$n$ is the maximum number of bodies


**If the number of bodies in current frame is less than n, then rest of the skeletal posture and temporal difference should be a matrix of 0's**

Dimensions of $S_i$ and $M_i$ are **(frame_l * joint_n * joint_d)**


## Mirror Augmentation
Augmentation is technique used to create artificial dataset from the given dataset. 

In mirror augmentation, we flip the given image horizontally and thus create a new dataset. Now before using the new dataset, we need to make sure that it is a valid data. In our case, it is straightforward, fliping the image horizontally does not change the action being performed. Hence, the new data is a valid one.


## Running the Model
To run the model use **main.ipynb**. Main.ipynb contains everything you need to run the model. It will import Data in an executable format, you can change the model according to your preferences (change max number of boldies as per your dataset). 

Main.ipynb will import your dataset using functions in utils.py, create the model, Get the data ready to be passed in to the model, train the model and test the results and print the classification report. It will also help to visualise mirror augmentation.

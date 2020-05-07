## Self Learning Car With Reinforcement Learning 

### Project Statement :
    Making a reinforcement learning agent(car) to travel around a city map and to reach from point A to Point B while taking the road and avoiding sand 


### Environment Description :
    Map : I have taken a city map. The map has roads and buildings. The roads are where the rl agent will learn to walk/drive. 
    Mask : We take the same city map black and white mask. Road is black and sand is white.
    Sand : sand is basically 90 deg (anti clock wise) rotated map of Mask. we will be using this sand map for training.

### Agent Description :
    Car : Car image is used as rl agent
    Action dimension : 1 (rotation)
    State dimension : 4 (image crop of the current location of size (100,100), distance between the agent and the goal, orientation, - - orientation)
    State dimension description :
        Image crop: Crop image of the car on the sand to tell the n/w that it is on sand or road
        Distance : euclidean distance between the agent and the goal point
        Orientation: Angle between the car axis and the goal 

### Algorithm Used :
    I have used Twin delayed DDPG Algorithm. I will add description about the algorithm later

### Rewards Strategy:
    This is the final strategy (i have added other things that i have tried in the bottom section)

    1. First train the model on the distance and without sand
    2. Once the model is mature enough and the rewards starts cumulating train the network on sand and without any destination
    3. Once both is trained add sand and destination. Also keep a living penalty to minimize the time

### Episode Ending Strategy :
    1. Episode is ended based on which part of the reward strategy is being done(described above)
    2. It gradually increases for the above steps (-4000, - 8000, -10000) 
    3. These negative values are cumulative gradients and are used to update done variable

### TD3 Architecture:
    I have used the same TD3 architecture and the image is encoded with the below convolution blocks

    ##############################
    # Total params: 6,272
    # Trainable params: 6,272
    # Non-trainable params: 0
    ##############################

    def ImageConv(in_dim, out_dim):
        model = nn.Sequential(
            nn.Conv2d(in_dim, 32, kernel_size=3,
                    stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),

            nn.Conv2d(32, 16, kernel_size=1, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(.2),

            nn.Conv2d(32, 16, kernel_size=1, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(9),
            nn.Flatten(),
            nn.Linear(16, out_dim, bias=False))
        return model





## Output Videos 

    #### Trained only for Distance:  

    ![Watch the video]()

    #### Trained only for Sand:  

    ![Watch the video]()

    #### Trained Sand + Distance:  

    ![Watch the video]()

## Things Done 

    1. Integrated TD3 with convolution
    2. Croped image of car on sand with angle 
    3. Training is happening partially for 3 different reward strategy (Rewards Strategy section)
     

### Challanges Faced and Possible solution

    1. kivy and pillow/numpy has different co-ordinate system
    2. Loading a pillow image and converting it to numpy tranposes the image. Because of  images were going wrong to the network
    3. Initially when i tried the network directly without following the reward strategy described above the car starts rotating. I was thinking because of numpy and pillow conversion images were wrong
    4. Later when i started following the reward strategy(step by step training) the rotation issue was fixed
    5. later realized the entire network trains based on the reward strategy and nothing much change is required on TD3
    6. I have tried different reward strategy and reward values
    7. Things i tried overall to stop rotation
        - used pretrained network embeddings(trained on the cropped images on BCE Loss sand vs road)
        - passed different orientation images 
        - passed images with and without other state variables like distance and orientation
        - tried changing max action values and temperature parameter change to stop the rotation issue 
        - also tried gradient clipping to stop rotation 
        

### ToDo:
    1. Need to plot the graph for sand/distance accumulated rewards to design the optimum training rewards

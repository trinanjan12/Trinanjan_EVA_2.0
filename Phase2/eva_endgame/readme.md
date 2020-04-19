# EVA ENDGAME PROJECT

## Project Description:

Train a Self learning car Env Using Twin Delay DDPG Algo

## Environment Description:
1. Observation sapce :
The env has 3 states. Image with car rotated, orientation, -orientataion. The orientataion is basically the angle between car velocity and the destination
2. Action sapce:
The action space dim is 1. which is the rotation of the car.

## Training Configuration:
1. The episode timestep is defined as 2000
2. after episode is done the car is respawned with a random (x,y) and the goals are also randomly changed
3. total 3 states are sent to the network, the image crop with car, and the 2 orientations 
4. The networks fills up replay buffer memory initially
5. After that policy model is  trained for each episode_steps

## Accomplishment :

1. Integrated car env with Twin Delay DDPG
2. Used image with car orientation instead of sensors
3. Also added orientation as 2 more state

## Current problems:
1. The network is not training well
2. Car starts rotating when the actor model predicts the same angle

## Things Tried:
1. First Tried with only image crop
2. Then orientation is also sent as state parameters
3. Done = true is made  based on cumulative reward (< -2000) to terminate episode
4. Different network architecture
5. Different max action and temperature values


## TODO:

1. Make a better documentation
2. Train with different parameter 
3. Train with better reward values and strategy
4. Make the code more modular and remove all unwanted commented sections
5. Figure out the role of temperature value
6. Make the image convolution network better
7. Train for more number of steps

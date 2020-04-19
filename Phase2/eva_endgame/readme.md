# EVA ENDGAME PROJECT

## Project Description:

Train a Self learning car Env Using Twin Delay DDPG Algo

## Environment Description:
1. Observation sapce :
The env has 3 states. Image with car rotated, orientation, -orientataion. The orientataion is basically the angle between car velocity and the destination
2. Action sapce:
The action space dim is 1. which is the rotation of the car.

## Training Configuration:

## Accomplishment :

1. Integrated car env with Twin Delay DDPG
2. Used image with car orientation instead of sensors
3. Also added orientation as 2 more state

## Current problems:
1. The network is not training well
2. Car starts rotating when the actor model predicts the same angle

## TODO:

1. Make a better documentation
2. Train with different parameter 
3. Train with better reward values and strategy
4. Make the code more modular and remove all unwanted commented sections
5. Figure out the role of temperature value
6. Make the image convolution network better
7. Train for more number of steps

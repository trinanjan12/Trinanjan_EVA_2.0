# EVA ENDGAME PROJECT

## This Code is unfinished and doesn't run yet

### Things that are done

Environment 
1. Removed all the sensor 
2. The environment has a function get_image() which returns image patch with car. 
we can crop the sand image along with the car using car position on the sand. Then apply Pillow rotate function to rotate the car image which shall be passed as an input to our brain.update

T3D code:
1. Actor model takes input image and convolutes. Finally predicts the action

2. Critic model : First some convolution operations are done on the input image to shrink it to a dimensions(something like embeddings) and after that the concatenation input of action and embeddings are feed to the critic network. Critic shall predict Q code


## TODO: 
1. Write train loop so that every time the TD3 update is called it should train the models and update them(based on policy freq)

2. Need to check whether the evaluation_policy class code could be reused 

3. Need to check if the input patch with direction is enough for the model to learn  or not. Otherwise need to send the direction information separately



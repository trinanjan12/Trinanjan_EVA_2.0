# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from PIL import Image as PILImage
from PIL import ImageDraw
from kivy.graphics.texture import Texture
import math

# Importing the Dqn object from our AI in ai.py
# from ai import Dqn
from final_t3d_new_pil import Train_TD3
# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Train_TD3('First_model_new_last_stage_sandincrease')

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '1429')
Config.set('graphics', 'height', '660')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

action2rotation = [0, 5, -5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")

total_reward = 0
episode_done = False

# Initializing the map
first_update = True
max_action_value = 5
count = 0
car_org_img = PILImage.open("./images/car.png").convert('RGBA')
car_org_img = car_org_img.resize((20, 10))

total_reward_end = -4000


sand_img_pil = PILImage.open("./images/test_1.png").convert('L')
def get_input_image(x, y, angle):
    global count
    crop_size = 50
    x, y, angle = int(x), int(y), int(-angle)

    car_rotated = car_org_img.rotate(angle, expand=1)
    sand_img_copy = sand_img_pil.copy()
    sand_img_copy.paste(car_rotated, (x, y), car_rotated)

    img_patch = sand_img_copy.crop(
        (x-crop_size, y-crop_size, x+crop_size, y+crop_size))

    # ##  FOR DEBUGGING
    # img_patch.save('./images_test/car_{}.png'.format(count))

    count += 1

    img_patch = img_patch  # (w,h)
    return img_patch




def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    sand = np.zeros((longueur, largeur))
    img = PILImage.open("./images/mask.png").convert('L')
    sand = np.asarray(img)/255
    goal_x = 1420
    goal_y = 622
    first_update = False
    global swap
    swap = 0


# Initializing the last distance
last_distance = 0

# Creating the car class


class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation


# Creating the game class
timesteps = 0


class Game(Widget):

    car = ObjectProperty(None)

    def serve_car(self):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)

    def update(self, dt):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global swap
        global total_reward
        global episode_done
        global timesteps
        global total_reward_end 

        # Width and height of the entire map 1429x660
        longueur = self.width
        largeur = self.height
        # set and load initial values
        if first_update:
            init()

        # distance in x and y from the goal
        # using this cordinate we can calculate the angle to move towards the target
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.
        # Input to the network
        # check = self.get_obs(xx,yy)

        last_signal = [[get_input_image(
            self.car.x, self.car.y, self.car.angle), (last_distance/(int(1574))), orientation, -orientation], episode_done]

        # Output from the network
        action = max_action_value * brain.update(last_reward, last_signal)

        # action = np.random.randint(-3, 3)
        rotation = int(action)
        self.car.move(rotation)

        # the distance for final goal
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        if timesteps < 50000:
            total_reward_end  = -5000
            if distance < last_distance:
                self.car.velocity = Vector(1.5, 0).rotate(self.car.angle)
                last_reward = .1
            else :
                last_reward = -.2
                self.car.velocity = Vector(.5, 0).rotate(self.car.angle)

        elif timesteps >= 50000 and timesteps < 150000 :
            total_reward_end  = -8000
            if sand[int(self.car.x), int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                last_reward = -.8
            else:  # otherwise
                self.car.velocity = Vector(1.5, 0).rotate(self.car.angle)
                last_reward = 0.5

        else : 
            total_reward_end  = -10000
            if sand[int(self.car.x), int(self.car.y)] > 0:
                self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
                #print(1, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))

                last_reward = -2
            # else car is on the road
            else:  # otherwise
                self.car.velocity = Vector(1.5, 0).rotate(self.car.angle)
                last_reward = -0.1
                #print(0, goal_x, goal_y, distance, int(self.car.x),int(self.car.y), im.read_pixel(int(self.car.x),int(self.car.y)))
                if distance < last_distance:
                    last_reward = 0.1

        # reset car if the car is near the wall
        distance_from_wall_x = 10
        if self.car.x < distance_from_wall_x:
            self.car.x = distance_from_wall_x
            last_reward = -1.5
        if self.car.x > self.width - distance_from_wall_x:
            self.car.x = self.width - distance_from_wall_x
            last_reward = -1.5
        if self.car.y < distance_from_wall_x:
            self.car.y = distance_from_wall_x
            last_reward = -1.5
        if self.car.y > self.height - distance_from_wall_x:
            self.car.y = self.height - distance_from_wall_x
            last_reward = -1.5

        episode_done = False
        total_reward += last_reward

        # check if the car has reached the goal and swap
        if total_reward < total_reward_end:
            print("----------Resetting Episode----------")
            episode_done = True
            total_reward = 0
            self.car.x = 353  # int(np.random.randint(25, self.width-25, 1)[0])
            self.car.y = 325  # int(np.random.randint(25, self.height-25, 1)[0])
            self.car.angle = 2  # initial
            print("----------new car location----------", self.car.x, self.car.y)

        if distance < 25:
            print("----------reached destination----------")

            if swap == 1:
                episode_done = True
                total_reward = 0
                goal_x = 1420
                goal_y = 622
                swap = 0
            else:
                goal_x = 9
                goal_y = 85
                swap = 1
            print("----------new goals are----------", goal_x, goal_y)
        last_distance = distance
        timesteps += 1


######### Running the whole thing #########


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        return parent


if __name__ == '__main__':
    CarApp().run()

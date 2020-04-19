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
from kivy.graphics.texture import Texture
import torchvision.transforms as transforms

# Importing the Dqn object from our AI in ai.py
from final_t3d import Train_TD3


######## Constants and Configs ########


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

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
brain = Train_TD3()
action2rotation = [0, 5, -5]
last_reward = 0
scores = []
im = CoreImage("./images/MASK1.png")
car_org_img = PILImage.open("./images/car.png").convert('RGBA')
car_org_img = car_org_img.resize((20, 10))
sand_img = PILImage.open("./images/mask.png").convert('L')

# Initializing the map
first_update = True

# Initializing the last distance
last_distance = 0
total_reward = 0

######## Helper Methods ########


def get_input_image(x, y, angle):
    x, y, angle = int(y), int(x), int(angle)
    # x, y, angle = 80, 80, 45
    car_rotated = car_org_img.rotate(angle, expand=1)
    sand_img_copy = sand_img.copy()
    sand_img_copy.paste(car_rotated, (x, y), car_rotated)

    img_patch = sand_img_copy.crop((x-30, y-30, x+30, y+30))
    img_patch = img_patch.convert('L')
    # img_patch.save('./output_patch/car.png')
    # sand_img_copy.save('./output_patch/sand.png')
    img_patch = np.array(img_patch) / 255
    # print(img_patch.min(),img_patch.max())
    return np.expand_dims(img_patch, axis=0)


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


########## Car Agent ##########


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

########## Car ENV ##########


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

        # Width and height of the entire map 1429x660
        longueur = self.width
        largeur = self.height
        # set and load initial values
        if first_update:
            init()

        episode_done = False
        # distance in x and y from the goal
        # using this cordinate we can calculate the angle to move towards the target
        xx = goal_x - self.car.x
        yy = goal_y - self.car.y

        # orientation gives you angle value towards direction
        # ?????????? basically orientation should be a angle between 2 points end , current
        ########## orientation = Vector(*self.car.pos).angle((goal_x,goal_y))/180.
        orientation = Vector(*self.car.velocity).angle((xx, yy))/180.

        # last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        # # networks predicts action which helps the cat to move
        # # rotation changes the car angle thus changes the position of the car
        # action = brain.update(last_reward, last_signal)
        # scores.append(brain.score())
        # rotation = action2rotation[action]

        # Input to the network
        last_signal = [[np.array(get_input_image(self.car.x, self.car.y, self.car.angle)), orientation, -orientation], episode_done]
        # Output from the network
        action = brain.update(last_reward, last_signal)
        # scores.append(brain.score())
        # action = np.random.randint(0, 3)
        rotation = int(action[0])
        # print(action)
        self.car.move(rotation)
        # the distance for final goal
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)

        # if the car position is on sand
        if sand[int(self.car.x), int(self.car.y)] > 0:
            self.car.velocity = Vector(0.5, 0).rotate(self.car.angle)
            # print(1, goal_x, goal_y, distance, int(self.car.x), int(
            #     self.car.y), im.read_pixel(int(self.car.x), int(self.car.y)))

            last_reward = -1

        # else car is on the road
        else:  # otherwise
            self.car.velocity = Vector(2, 0).rotate(self.car.angle)
            last_reward = -0.2
            # print(0, goal_x, goal_y, distance, int(self.car.x), int(
            #     self.car.y), im.read_pixel(int(self.car.x), int(self.car.y)))
            if distance < last_distance:
                last_reward = 0.1

        # reset car if the car is near the wall
        if self.car.x < 5:
            self.car.x = 5
            last_reward = -1
        if self.car.x > self.width - 5:
            self.car.x = self.width - 5
            last_reward = -1
        if self.car.y < 5:
            self.car.y = 5
            last_reward = -1
        if self.car.y > self.height - 5:
            self.car.y = self.height - 5
            last_reward = -1

        # check if the car has reached the goal and swap
        if distance < 25 or total_reward < -2000:
            episode_done = True
            total_reward = 0
            self.car.x = int(np.random.randint(25, self.width-25, 1)[0])
            self.car.y = int(np.random.randint(25, self.height-25, 1)[0])
            goal_x = int(np.random.randint(25, self.width-25, 1)[0])
            goal_y = int(np.random.randint(25, self.width-25, 1)[0])
            print("new episode goals", (self.car.x, self.car.y), (goal_x, goal_y))
            # if swap == 1:
            #     goal_x = int(np.random.randint(25, self.width-25, 1)[0])
            #     goal_y = int(np.random.randint(25, self.width-25, 1)[0])
            #     swap = 0
            # else:
            #     goal_x = int(np.random.randint(25, self.width-25, 1)[0])
            #     goal_y = int(np.random.randint(25, self.width-25, 1)[0])
            #     swap = 1
        last_distance = distance
        total_reward += last_reward

######### Adding the painting tools #########


class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8, 0.7, 0)
            d = 10.
            touch.ud['line'] = Line(points=(touch.x, touch.y), width=10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x), int(touch.y)] = 1
            img = PILImage.fromarray(sand.astype("uint8")*255)
            img.save("./images/sand.jpg")

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10: int(touch.x) + 10,
                 int(touch.y) - 10: int(touch.y) + 10] = 1

            last_x = x
            last_y = y

######### Running the whole thing #########


class CarApp(App):

    def build(self):
        parent = Game()
        parent.serve_car()
        Clock.schedule_interval(parent.update, 1.0/60.0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text='clear')
        savebtn = Button(text='save', pos=(parent.width, 0))
        loadbtn = Button(text='load', pos=(2 * parent.width, 0))
        clearbtn.bind(on_release=self.clear_canvas)
        savebtn.bind(on_release=self.save)
        loadbtn.bind(on_release=self.load)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        return parent

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur, largeur))

    def save(self, obj):
        print("saving brain...")
        # brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        # brain.load()


if __name__ == '__main__':
    CarApp().run()

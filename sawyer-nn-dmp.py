#!/usr/bin/env python

import math
import rospy
import cv2 
import intera_interface
from pyrdmp.dmp import DynamicMovementPrimitive as DMP
from pyrdmp.plots import *
from pyrdmp.utils import *

from keras.models import load_model
from SawyerClass import Sawyer

BLUELOWER = np.array([110, 100, 100])
BLUEUPPER = np.array([120, 255, 255])

# Determines noise clear for morph
KERNELOPEN = np.ones((5, 5))
KERNELCLOSE = np.ones((5, 5))

# Font details for display windows
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)


def transform(x_p, y_p, x_robot, y_robot, x_image, y_image):

    a_y = (y_robot[0]-y_robot[1])/(y_image[1]-y_image[0])
    b_y = y_robot[1]-a_y*y_image[0]
    y_r = a_y*y_p+b_y
    
    a_x = (x_robot[0]-x_robot[1])/(x_image[1]-x_image[0])
    b_x = x_robot[1]-a_x*x_image[0]
    x_r = a_x*x_p+b_x

    return [x_r, y_r]


def detection():

    cam = cv2.VideoCapture(-1)

    print(cam.isOpened())

    cameraMatrix = np.array([[506.857008, 0.000000, 311.541447], [0.000000, 511.072198, 257.798417], [0.000000, 0.000000, 1.000000]])
    distCoeffs = np.array([0.047441, -0.104070, 0.006161, 0.000338, 0.000000])

    y_robot = [-0.4, -0.8]
    y_image = [173, 355]

    x_robot = [-0.3, 0.3]
    x_image = [178, 448]
    
    positions = []
    
    for i in range(5):

        ret_val, img = cam.read()
        if not ret_val: continue
        height, width, channels = img.shape

        und_img = cv2.undistort(img, cameraMatrix, distCoeffs)

        cv2.line(und_img, (x_image[1], y_image[0]), (x_image[0], y_image[0]), (0, 0, 255), 1)
        cv2.line(und_img, (x_image[0], y_image[0]), (x_image[0], y_image[1]), (0, 0, 255), 1)
        cv2.line(und_img, (x_image[0], y_image[1]), (x_image[1], y_image[1]), (0, 0, 255), 1)
        cv2.line(und_img, (x_image[1], y_image[1]), (x_image[1], y_image[0]), (0, 0, 255), 1)

        # Convert image to HSV
        imHSV = cv2.cvtColor(und_img, cv2.COLOR_BGR2HSV)

        # Threshold the colors  
        mask_blue = cv2.inRange(imHSV, BLUELOWER, BLUEUPPER)
        mask_blue_open = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, KERNELOPEN)
        mask_blue_close = cv2.morphologyEx(mask_blue_open, cv2.MORPH_CLOSE, KERNELCLOSE)

        conts, hierarchy = cv2.findContours(mask_blue_close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # Hold the centers of the detected objects
        location = []

        # Loop over the contours
        for c in conts:

            # compute the center of the contour
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            cv2.circle(und_img, (cX, cY), 1, (0, 0, 255), -1)

            location.append([cX, cY])

        for c in location:

            dummy = transform(c[0], c[1], x_robot, y_robot, x_image, y_image)
            positions.append(dummy)

        print(positions)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit

    return positions


# Define  new node
rospy.init_node("Sawyer_DMP")

# Create an object to interface with the arm
limb = intera_interface.Limb('right')

# Create an object to interface with the gripper
gripper = intera_interface.Gripper('right')

# Models location
forward_model_file = '/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/MyScripts/sawyer-nn-pyrdmp/weights/ForwardModel/4DOF/forward.h5'
inverse_model_file = '/home/michail/ros_ws/src/intera_sdk/intera_examples/scripts/MyScripts/sawyer-nn-pyrdmp/weights/InverseModel/MLP_2.h5'

# Load the models
forwardModel = load_model(forward_model_file)
inverseModel = load_model(inverse_model_file)

# Move the robot to the starting point
angles = limb.joint_angles()
angles['right_j0'] = math.radians(0)
angles['right_j1'] = math.radians(-50)
angles['right_j2'] = math.radians(0)
angles['right_j3'] = math.radians(120)
angles['right_j4'] = math.radians(0)
angles['right_j5'] = math.radians(0)
angles['right_j6'] = math.radians(0)
limb.move_to_joint_positions(angles)

# Get the position of the cube
print('Acquiring Target')
target = detection()

print('Target found at:')
target = np.array([target[0][0], target[0][1], -0.04])
print(target)

# Get the initial position of the robot
joint_positions = limb.joint_angles()

# Just a vector to name the joints of the robot
joint_names = ['right_j0', 'right_j1', 'right_j3', 'right_j5']
full_names = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

q = np.array([[float(joint_positions[i]) for i in joint_names]])

# Damping factor
d = np.array([100, 100, 100, 100])

# Declare the joint position history and time history
t = np.array([[0]])
q_demo = np.empty((0, 4))

# Initialize some counters
counter = 0
error = 1000
convert = 1000000000000
thresh = -0.06

while error > 0.075:

    counter = counter + 1

    # Accumulate the time vector and the joint history
    time = rospy.Time.now()
    dt = float(time.secs)/float(convert)
    t = np.append(t, t[counter-1]+dt)
    q_demo = np.vstack((q_demo, [q[0][0], q[0][1], q[0][2], q[0][3]]))

    # Perform the forward model prediction 
    x = np.divide(forwardModel.predict(q), 100)

    # Transform the prediction
    x_e = np.multiply(x-target, 1000)

    # Based on the forward model prediction, predict the next motor command
    new_q = np.radians(inverseModel.predict(x_e))

    # Send the velocity command to the robot
    dq = limb.joint_angles()
    dq['right_j0'] = d[0]*new_q[0][0]
    dq['right_j1'] = d[1]*new_q[0][1]
    dq['right_j2'] = 0
    dq['right_j3'] = d[2]*new_q[0][2]
    dq['right_j4'] = 0
    dq['right_j5'] = d[3]*new_q[0][3]
    dq['right_j6'] = 0
    limb.set_joint_velocities(dq)

    # Get the new state of the robot
    joint_positions = limb.joint_angles()
    q_r = np.array([[float(joint_positions[i]) for i in joint_names]])
    q = np.array([[q_r[0][0], q_r[0][1], q_r[0][2], q_r[0][3]]])

    # Find the error from the target
    error = math.fabs(x[0][2]-thresh)

# Initialize the DMP class
my_dmp = DMP(20, 20, 0)

# Filtering Parameters
window = 5
blends = 10

# Normalize the time vector
t = t[:-1]
t = normalize_vector(t)

# Get the phase from the time vector
s = my_dmp.phase(t)

# Get the Gaussian
psv = my_dmp.distributions(s)

# Compute velocity and acceleration
dq_demo = np.zeros(q_demo.shape)

for i in range(0, q_demo.shape[1]):
    q_demo[:, i] = smooth_trajectory(q_demo[:, i], window)
    dq_demo[:, i] = vel(q_demo[:, i], t)

# Filter the position velocity and acceleration signals
f_q = np.zeros(q_demo.shape)
f_dq = np.zeros(q_demo.shape)
f_ddq = np.zeros(q_demo.shape)

for i in range(0, q_demo.shape[1]):
    f_q[:, i] = blend_trajectory(q_demo[:, i], dq_demo[:, i], t, blends)
    f_dq[:, i] = vel(f_q[:, i], t)
    f_ddq[:, i] = vel(f_dq[:, i], t)

# Imitation Learning
ftarget = np.zeros(q_demo.shape)
w = np.zeros((my_dmp.ng, q_demo.shape[1]))

print('Imitation start')

for i in range(0, q.shape[1]):
    ftarget[:, i], w[:, i] = my_dmp.imitate(f_q[:, i], f_dq[:, i], f_ddq[:, i], t, s, psv)

print('Imitation done')

# Generate the Learned trajectory
x = np.zeros(q_demo.shape)
dx = np.zeros(q_demo.shape)
ddx = np.zeros(q_demo.shape)

for i in range(0, q.shape[1]):
    ddx[:, i], dx[:, i], x[:, i] = my_dmp.generate(w[:, i], f_q[0, i], f_q[-1, i], t, s, psv)

# Adapt using Reinforcement Learning
x_r = np.zeros(q_demo.shape)
dx_r = np.zeros(q_demo.shape)
ddx_r = np.zeros(q_demo.shape)

# Call the Sawyer Class
robot = Sawyer()

# First find the target in joint space
orientation = [180, 0, 90]
coordinates = [target[0], target[1], target[2]]

# Call the IK method
ik = robot.Inverse_Kinematics(coordinates,orientation)

# Move the robot again to the starting point
angles = limb.joint_angles()
angles['right_j0']=math.radians(0)
angles['right_j1']=math.radians(-50)
angles['right_j2']=math.radians(0)
angles['right_j3']=math.radians(120)
angles['right_j4']=math.radians(0)
angles['right_j5']=math.radians(0)
angles['right_j6']=math.radians(0)
limb.move_to_joint_positions(angles)

# Define the target joint positions
joint_target = np.array([[float(ik[i]) for i in joint_names]])
goal = np.array([[joint_target[0][0], joint_target[0][1], joint_target[0][2], joint_target[0][3]]])

print('Adaptation start')
samples = 10
rate = 0.5

print(goal)

for i in range(0, q_demo.shape[1]):
    ddx_r[:, i], dx_r[:, i], x_r[:, i] = my_dmp.adapt(w[:, i], x[0, i], goal[0][i], t, s, psv, samples, rate)

print('Adaptation complete')

# Plot functions
comparison(t, f_q, x, x_r)
show_all()



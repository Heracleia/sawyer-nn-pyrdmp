#!/usr/bin/env python2

import math
import rospy
import cv2
import intera_interface
from pyrdmp.dmp import DynamicMovementPrimitive as DMP
from pyrdmp.plots import *
from pyrdmp.utils import *

from keras.models import load_model

from geometry_msgs.msg import Pose
from sawyer_robot import Sawyer

BLUELOWER = np.array([110, 100, 100])
BLUEUPPER = np.array([120, 255, 255])

# Determines noise clear for morph
KERNELOPEN = np.ones((5, 5))
KERNELCLOSE = np.ones((5, 5))

# Font details for display windows
FONTFACE = cv2.FONT_HERSHEY_SIMPLEX
FONTSCALE = 1
FONTCOLOR = (255, 255, 255)

# Camera calibration values - Specific to C930e 
CAMERAMATRIX = np.array([[506.857008, 0.000000, 311.541447], 
                         [0.000000, 511.072198, 257.798417], 
                         [0.000000, 0.000000, 1.000000]])
DISTORTION = np.array([0.047441, -0.104070, 0.006161, 0.000338, 0.000000])

CARTIM = [[162, 440], [212, 399]] #[[178, 448], [173, 355]]  # [[XX],[YY]] of the calibration points on table
CARTBOT = [[-0.3,0.3], [-0.4,-0.8]] # [[XX],[YY]] for the cartesian EE table values
GOAL = [600,300] # Drop off point of cylinders
ZLOW = -0.065 # Pick up height
ZHIGH = 0.26 # Drop off height (to reach over lip of box)

# Models location
forward_model_file = 'weights/ForwardModel/7DOF/UFMD7_5M.h5'
inverse_model_file = 'weights/InverseModel/MLP_2.h5'

# Filters blocks out of image and returns a list of x-y pairs in relation to the end-effector
def detectBlock(cap):
    for i in range(5): cap.grab() # Disregard old frames

    ret_val, im = cap.read()
    while not ret_val: # In case an image is not captured
        ret_val, im = cap.read()

    und_im = cv2.undistort(im, CAMERAMATRIX, DISTORTION) # Remove distortions
    imHSV = cv2.cvtColor(und_im, cv2.COLOR_BGR2HSV)
        
    mask = cv2.inRange(imHSV, BLUELOWER, BLUEUPPER) # Masking out blue cylinders
    mask_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, KERNELOPEN)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, KERNELCLOSE)
    
    _, conts, h = cv2.findContours(mask_close.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(und_im, conts, -1, (255, 255, 0), 1) # Helpful for visualization
        
    centers = [getCenter(*cv2.boundingRect(c)) for c in conts] # Calc center of each cylinder
    return [pixelsToCartesian(*c) for c in centers] # Return centers (in cartesian instead of pixels)

# Returns center of block based on bounding box
def getCenter(x, y, w, h):
    return (((int)(x + 0.5*w)), ((int)(y + 0.5*h)))

# Returns x,y coordinates based on linear relationship to pixel values.
def pixelsToCartesian(cx, cy):
    a_y = (CARTBOT[1][0]-CARTBOT[1][1])/(CARTIM[1][1]-CARTIM[1][0])
    b_y = CARTBOT[1][1]-a_y*CARTIM[1][0]
    y = a_y*cy+b_y
    a_x = (CARTBOT[0][0]-CARTBOT[0][1])/(CARTIM[0][1]-CARTIM[0][0])
    b_x = CARTBOT[0][1]-a_x*CARTIM[0][0]
    x = a_x*cx+b_x
    return (x,y)


# Define  new node
rospy.init_node("Sawyer_DMP")

# Create an object to interface with the arm
limb = intera_interface.Limb('right')

# Create an object to interface with the gripper
gripper = intera_interface.Gripper('right')

cap = cv2.VideoCapture(-1)
if not cap.isOpened():
    exit(1)

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
target = detectBlock(cap)

print(target)

print('Target found at:')
target = np.array([target[0][0], target[0][1], -0.04])
print(target)

# Get the initial position of the robot
joint_positions = limb.joint_angles()
q = np.array([[float(joint_positions[i]) for i in limb.joint_names()]])

# Damping factor
d = np.array([100, 100, 100, 100])

# Declare the joint position history and time history
recorded_t = []
recorded_q = []

error = 1000
convert = 1000000000000
thresh = -0.06
dq = limb.joint_angles()

while error > 0.075:

    # Accumulate the time vector and the joint history
    recorded_t.append(rospy.get_time())

    # Perform the forward model prediction 
    x = forwardModel.predict(q)/100

    # Perform the forward model prediction 
    x_e = 1000*(x - target)

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
    q = np.array([[float(joint_positions[i]) for i in limb.joint_names()]])
    recorded_q.append(q)

    # Find the error from the target
    error = math.fabs(x[0][2]-thresh)


#TODO: Perhaps need convert
t = np.array(recorded_t) - recorded_t[0]
q_demo = np.concatenate(recorded_q)

# Initialize the DMP class
my_dmp = DMP(20, 20, 0)

# Filtering Parameters
window = 5
blends = 10

# Normalize the time vector
#t = t[:-1] #TODO: 
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

# First find the target in joint space
orientation = [180, 0, 90]
coordinates = [target[0], target[1], target[2]]

robot = Sawyer()
robot_ik = robot.inverse_kinematics(limb, coordinates, orientation)

# Move the robot again to the starting point
angles = limb.joint_angles()
angles['right_j0'] = math.radians(0)
angles['right_j1'] = math.radians(-50)
angles['right_j2'] = math.radians(0)
angles['right_j3'] = math.radians(120)
angles['right_j4'] = math.radians(0)
angles['right_j5'] = math.radians(0)
angles['right_j6'] = math.radians(0)
limb.move_to_joint_positions(angles)

print(robot_ik)

# Define the target joint positions
goal = np.array([[float(robot_ik[i]) for i in limb.joint_names()]])

print('Adaptation start')
samples = 10
rate = 0.5

print(goal)

for i in range(0, q_demo.shape[1]):
    ddx_r[:, i], dx_r[:, i], x_r[:, i], _ = my_dmp.adapt(w[:, i], x[0, i], goal[0][i], t, s, psv, samples, rate)

print('Adaptation complete')
print(x_r[-1])

# Plot functions
comparison(t, f_q, x, x_r)
show_all()

# Save trajectory
traj_final = np.concatenate((x_r, np.multiply(np.ones((x_r.shape[0], 1)), 0.0402075604203)), axis=1)

# Create the trajectory file
max_time = t[-1]
time = np.linspace(0, max_time, x_r.shape[0]).reshape((x_r.shape[0], 1))

traj_final = np.concatenate((t.reshape((-1, 1)), traj_final), axis=1)

# Save trajectory
np.savetxt('traj_final.txt', traj_final, delimiter=',', header='time,right_j0,right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_gripper', comments='', fmt="%1.12f")



"""
interp = 200

# Create the trajectory file
max_time = t[-1]
time = np.linspace(0, max_time, interp)
new_q = np.array([np.interp(np.linspace(0, len(x_r)-1, interp), np.linspace(0, len(x_r)-1, len(x_r)), x_r[:, i]) for i in range(len(x_r[0]))])

traj_final = np.concatenate((new_q.T, np.multiply(np.ones((interp, 1)), 0.0402075604203)), axis=1)

traj_final = np.concatenate((time.reshape((-1, 1)), traj_final), axis=1)

# Save trajectory
np.savetxt('traj_final.txt', traj_final, delimiter=',', header='time,right_j0,right_j1,right_j2,right_j3,right_j4,right_j5,right_j6,right_gripper', comments='', fmt="%1.12f")

"""
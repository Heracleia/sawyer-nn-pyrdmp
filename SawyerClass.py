#!/usr/bin/env python

# Updated Sawyer Robot Class by Michail Theofanidis

#import libraries
import os
import numpy as np
import sympy as sp
import math
import rospy
import roslib
import tf
import geometry_msgs.msg
import intera_interface 

from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from geometry_msgs.msg import (
    PoseStamped,
    Pose,
    Point,
    Quaternion,
)
from std_msgs.msg import Header
from sensor_msgs.msg import JointState

from intera_core_msgs.srv import (
    SolvePositionIK,
    SolvePositionIKRequest,
)


# Decleration of the Sawyer Robot Class
class Sawyer:
    def __init__(self):

        # Number of joints, links and offsets
        self.num_joints = 7
        self.num_links= 8
        self.num_offsets= 7

        # Arrays that store the link parameters and offset parameters
        self.L = np.array([0.0794732, 0.237, 0.142537, 0.259989, 0.126442, 0.274653, 0.105515, 0.0695])
        self.d = np.array([0.0814619, 0.0499419, 0.140042, 0.0419592, 0.1224936, 0.031188, 0.109824])

        # Joint Names array
        self.joint_names = ['right_j%i' % i for i in range(self.num_joints)]

        # Joint variable
        self.q = [sp.Symbol('q%i' % i) for i in range(self.num_joints)]

        # Homogeneous Transformation Matrices of the Robot
        self.T_R_01 = sp.Matrix([[sp.cos(self.q[0]), -sp.sin(self.q[0]), 0, 0],
         [sp.sin(self.q[0]), sp.cos(self.q[0]), 0, 0],
         [0, 0, 1, self.L[0]],
         [0, 0, 0, 1]])

        self.T_R_12 = sp.Matrix([[sp.cos(self.q[1]), -sp.sin(self.q[1]), 0, self.d[0]],
         [0, 0, 1, self.d[1]],
         [-sp.sin(self.q[1]), -sp.cos(self.q[1]), 0, self.L[1]],
         [0, 0, 0, 1]])

        self.T_R_23 = sp.Matrix([[sp.cos(self.q[2]), -sp.sin(self.q[2]), 0, 0],
         [0, 0, -1, -self.d[2]],
         [sp.sin(self.q[2]), sp.cos(self.q[2]), 0, self.L[2]],
         [0, 0, 0, 1]])

        self.T_R_34 = sp.Matrix([[sp.cos(self.q[3]), -sp.sin(self.q[3]), 0, 0],
         [0, 0, 1, -self.d[3]],
         [-sp.sin(self.q[3]), -sp.cos(self.q[3]), 0, self.L[3]],
         [0, 0, 0, 1]])

        self.T_R_45 = sp.Matrix([[sp.cos(self.q[4]), -sp.sin(self.q[4]), 0, 0],
         [0, 0, -1, -self.d[4]],
         [sp.sin(self.q[4]), sp.cos(self.q[4]), 0, -self.L[4]],
         [0, 0, 0, 1]])

        self.T_R_56 = sp.Matrix([[sp.cos(self.q[5]), -sp.sin(self.q[5]), 0, 0],
         [0, 0, 1, self.d[5]],
         [-sp.sin(self.q[5]), -sp.cos(self.q[5]), 0, self.L[5]],
         [0, 0, 0, 1]])

        self.T_R_67 = sp.Matrix([[sp.cos(self.q[6]), -sp.sin(self.q[6]), 0, 0],
         [0, 0, -1, -self.d[6]],
         [sp.sin(self.q[6]), sp.cos(self.q[6]), 0, self.L[6]],
         [0, 0, 0, 1]])

        self.T_R_89 = sp.Matrix([[0,-1, 0, 0],
         [1, 0, 0, 0],
         [0, 0, 1, 0.0245],
         [0, 0, 0, 1]])

        self.T_R_9e = sp.Matrix([[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 1, 0.0450],
         [0, 0, 0, 1]])

        # Trasformation Tree of the Robot
        self.T_R=[self.T_R_01, self.T_R_12, self.T_R_23, self.T_R_34, self.T_R_45, self.T_R_56, self.T_R_67, self.T_R_89, self.T_R_9e]

    # Forward Kinematics for the Robot
    def get_T_f(self):

        self.Tf_R=self.Forward_Kinematics(self.T_R)

        return self.Tf_R

    # Method to adjust the offset in joint space
    def JointOffset(self,angles):

        angles[1]=angles[1]+math.radians(90)
        angles[6]=angles[6]+math.radians(170)+math.radians(90)

        return angles

    # Function that performs the Forward Kinematic Equations
    def Forward_Kinematics(self, trans):


        self.temp=[trans[0]]
        counter=-1;

        # Traverse through the transformation tree
        for i in trans[1:]:#

            counter=counter+1
            self.temp.append(self.temp[counter]*i)#
  
        return self.temp

    # Function that return the IK solution given a position and orientation
    def Inverse_Kinematics(self,coordinates,orientation):

        # Define  new node
        #rospy.init_node("Sawyer_ik_client")

        # Create an object to interface with the arm
        Robot_limb=intera_interface.Limb('right')

        # Find the Equivalent gamma (yaw), beta (pitch) and alpha (roll)
        quaternion = tf.transformations.quaternion_from_euler(math.radians(orientation[0]), math.radians(orientation[1]), math.radians(orientation[2]))

        # Get the current state of the robot
        angles=Robot_limb.joint_angles()

        # Initialize the IK service
        limb = "right"
        ns = "ExternalTools/" + limb + "/PositionKinematicsNode/IKService"
        iksvc = rospy.ServiceProxy(ns, SolvePositionIK)
        ikreq = SolvePositionIKRequest()
        hdr = Header(stamp=rospy.Time.now(), frame_id='base')

        # Define the pose
        poses = {
            'right': PoseStamped(
                header=hdr,
                pose=Pose(
                    position=Point(
                        x=coordinates[0],
                        y=coordinates[1],
                        z=coordinates[2],
                        ),
                    orientation=Quaternion(
                        x=quaternion[0],
                        y=quaternion[1],
                        z=quaternion[2],
                        w=quaternion[3],
                        ),
                    ),
                ),
            }

        # Add desired pose for inverse kinematics
        ikreq.pose_stamp.append(poses[limb])

        # Request inverse kinematics from base to "right_hand" link
        ikreq.tip_names.append('right_hand')

        # The joint seed is where the IK position solver starts its optimization
        ikreq.seed_mode = ikreq.SEED_USER

        joint_seed=[None]*7
        joint_seed[0]=angles['right_j0']
        joint_seed[1]=angles['right_j1']
        joint_seed[2]=angles['right_j2']
        joint_seed[3]=angles['right_j3']
        joint_seed[4]=angles['right_j4']
        joint_seed[5]=angles['right_j5']
        joint_seed[6]=angles['right_j6']

        seed = JointState()
        seed.name = ['right_j0', 'right_j1', 'right_j2', 'right_j3', 'right_j4', 'right_j5', 'right_j6']

        seed.position =joint_seed
        ikreq.seed_angles.append(seed)

        # Optimize the null space in terms of joint configuration
        ikreq.use_nullspace_goal.append(True)
        # The nullspace goal can either be the full set or subset of joint angles
        goal = JointState()
        goal.name = ['right_j2']
        goal.position = [0]
        ikreq.nullspace_goal.append(goal)

        # Define the gain of the gradient descent essentially
        ikreq.nullspace_gain.append(0.4)

        # Get the response from the server
        try:
            rospy.wait_for_service(ns, 5.0)
            resp = iksvc(ikreq)
        except (rospy.ServiceException, rospy.ROSException), e:
            rospy.logerr("Service call failed: %s" % (e,))

        # Check if result valid, and type of seed ultimately used to get solution
        if (resp.result_type[0] > 0):
            seed_str = {
                        ikreq.SEED_USER: 'User Provided Seed',
                        ikreq.SEED_CURRENT: 'Current Joint Angles',
                        ikreq.SEED_NS_MAP: 'Nullspace Setpoints',
                       }.get(resp.result_type[0], 'None')
            rospy.loginfo("SUCCESS - Valid Joint Solution Found from Seed Type: %s" %
                  (seed_str,))
            # Format solution into Limb API-compatible dictionary
            limb_joints = dict(zip(resp.joints[0].name, resp.joints[0].position))
            rospy.loginfo("\nIK Joint Solution:\n%s", limb_joints)
            rospy.loginfo("------------------")
            rospy.loginfo("Response Message:\n%s", resp)
        else:
            rospy.logerr("INVALID POSE - No Valid Joint Solution Found.")
            rospy.logerr("Result Error %d", resp.result_type[0])

        angles['right_j0']=limb_joints['right_j0']
        angles['right_j1']=limb_joints['right_j1']
        angles['right_j2']=limb_joints['right_j2']
        angles['right_j3']=limb_joints['right_j3']
        angles['right_j4']=limb_joints['right_j4']
        angles['right_j5']=limb_joints['right_j5']
        angles['right_j6']=limb_joints['right_j6']

        return angles




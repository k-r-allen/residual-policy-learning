from .controllers import StatelessController, StateMachineController

import numpy as np
# import pybullet as p


DEBUG = False


def move_controller_fn(observation, target_position, atol=1e-3, gain=10.):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    current_position = observation['observation'][:3]

    # Check if we're already near the target
    position_error = np.sum(np.subtract(current_position, target_position)**2)
    done = position_error < atol

    action = gain * np.subtract(target_position, current_position)
    action = np.hstack((action, 0.))

    return action, done

def open_gripper_controller_fn(observation, atol=1e-3):
    """
    Open a two-fingered gripper all the way.
    """
    return 1., True

def close_gripper_controller_fn(observation, atol=1e-3):
    """
    Close a two-fingered gripper.
    """
    return -1., True

def create_pick_and_place_controller(atol=1e-3):
    """
    Create a controller for grasping an object with a two-fingered gripper.

    Returns
    -------
    grasp_controller : StateMachineController
    """


    # First move gripper above the graspee
    def get_above_target_pose():
        graspee_position, graspee_orientation = p.getBasePositionAndOrientation(graspee_id, physicsClientId=physics_client_id)
        above_graspee_position = [ x for x in graspee_position ]
        above_graspee_position[2] = workspace_height
        return p.multiplyTransforms(above_graspee_position, graspee_orientation, relative_grasp_position, relative_grasp_orientation)
    fn = lambda obs : move_controller_fn(obs, robot_id, end_effector_id, get_above_target_pose, joint_indices, free_joint_map,
        atol=atol, physics_client_id=physics_client_id)
    move_controller1 = StatelessController(fn)

    # Open the gripper
    fn = lambda obs: open_gripper_controller_fn(obs, robot_id, left_finger_id, right_finger_id, free_joint_map,
        physics_client_id=physics_client_id)
    open_gripper_controller = StatelessController(fn)

    # Move down to grasp
    def get_target_pose():
        graspee_position, graspee_orientation = p.getBasePositionAndOrientation(graspee_id, physicsClientId=physics_client_id)
        return p.multiplyTransforms(graspee_position, graspee_orientation, relative_grasp_position, relative_grasp_orientation)
    fn = lambda obs : move_controller_fn(obs, robot_id, end_effector_id, get_target_pose, joint_indices, free_joint_map,
        atol=atol, physics_client_id=physics_client_id)
    move_controller2 = StatelessController(fn)

    # Close the gripper
    fn = lambda obs: close_gripper_controller_fn(obs, robot_id, left_finger_id, right_finger_id, free_joint_map,
        physics_client_id=physics_client_id)
    close_gripper_controller = StatelessController(fn)

    # Move back up with the grasped object
    fn = lambda obs : move_controller_fn(obs, robot_id, end_effector_id, get_above_target_pose, joint_indices, free_joint_map,
        atol=atol, physics_client_id=physics_client_id)
    move_controller3 = StatelessController(fn)

    # Check joint velocities to identify when the robot has stalled
    def is_stalled():
        stalled = True
        for joint_idx in joint_indices:
            joint_vel = p.getJointState(robot_id, joint_idx, physicsClientId=physics_client_id)[1]
            if np.sum(np.square(joint_vel)) > atol:
                stalled = False
                break
        return stalled

    # A state machine controller is just a sequence of controllers
    controller_sequence = [move_controller1, open_gripper_controller, move_controller2, close_gripper_controller, 
        move_controller3]

    return StateMachineController(controller_sequence, is_stalled=is_stalled)

def create_pick_and_place_controller_from_obs(env, relative_grasp_position=(0., 0., -0.02)):
    physics_client_id = env.physics_client_id

    robot_id, left_end_effector_id = env.get_object_id('left_endpoint') 
    _, left_left_finger_id = env.get_object_id('l_gripper_l_finger_joint')
    _, left_right_finger_id = env.get_object_id('l_gripper_r_finger_joint')
    graspee_id = env.get_object_id('block')

    free_joint_map = {joint_idx : idx for (idx, joint_idx) in enumerate(env.free_joints)}

    return create_grasp_controller(robot_id, left_end_effector_id, left_left_finger_id, left_right_finger_id, 
        graspee_id, free_joint_map, relative_grasp_position, physics_client_id=physics_client_id)


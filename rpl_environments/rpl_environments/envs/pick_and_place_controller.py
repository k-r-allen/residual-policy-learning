import numpy as np

DEBUG = False

def get_move_action(observation, target_position, atol=1e-3, gain=10., close_gripper=False):
    """
    Move an end effector to a position and orientation.
    """
    # Get the currents
    current_position = observation['observation'][:3]

    action = gain * np.subtract(target_position, current_position)
    if close_gripper:
        gripper_action = -1.
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action

def block_is_grasped(obs, relative_grasp_position, atol=1e-3):
    return block_inside_grippers(obs, relative_grasp_position, atol=atol) and grippers_are_closed(obs, atol=atol)

def block_inside_grippers(obs, relative_grasp_position, atol=1e-3):
    gripper_position = obs['observation'][:3]
    block_position = obs['observation'][3:6]

    relative_position = np.subtract(gripper_position, block_position)

    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol

def grippers_are_closed(obs, atol=1e-3):
    gripper_state = obs['observation'][9:11]
    return abs(gripper_state[0] - 0.024) < atol

def grippers_are_open(obs, atol=1e-3):
    gripper_state = obs['observation'][9:11]
    return abs(gripper_state[0] - 0.05) < atol

def get_pick_and_place_control(obs, relative_grasp_position=(0., 0., -0.02), workspace_height=0.1, atol=1e-3, gain=10):
    """
    Returns
    -------
    action : [float] * 4
    """
    gripper_position = obs['observation'][:3]
    block_position = obs['observation'][3:6]
    place_position = obs['desired_goal']

    # If the gripper is already grasping the block
    if block_is_grasped(obs, relative_grasp_position, atol=atol):

        # If the block is already at the place position, do nothing except keep the gripper closed
        if np.sum(np.subtract(block_position, place_position)**2) < atol:
            if DEBUG:
                print("The block is already at the place position; do nothing")
            return np.array([0., 0., 0., -1.])

        # Move to the place position while keeping the gripper closed
        target_position = np.add(place_position, relative_grasp_position)
        target_position[2] += workspace_height/2.
        if DEBUG:
            print("Move to above the place position")
        return get_move_action(obs, target_position, atol=atol, close_gripper=True, gain=gain)

    # If the block is ready to be grasped
    if block_inside_grippers(obs, relative_grasp_position, atol=atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.])

    # If the gripper is above the block
    if (gripper_position[0] - block_position[0])**2 + (gripper_position[1] - block_position[1])**2 < atol:

        # If the grippers are closed, open them
        if not grippers_are_open(obs, atol=atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., 1.])

        # Move down to grasp
        target_position = np.add(block_position, relative_grasp_position)
        if DEBUG:
            print("Move down to grasp")
        return get_move_action(obs, target_position, atol=atol, gain=gain)


    # Else move the gripper to above the block
    target_position = np.add(block_position, relative_grasp_position)
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the block")
    return get_move_action(obs, target_position, atol=atol, gain=gain)



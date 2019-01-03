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

def block_is_grasped(obs, gripper_position, block_position, relative_grasp_position, atol=1e-3):
    block_inside = block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=atol) 
    grippers_closed = grippers_are_closed(obs, atol=atol)

    return block_inside and grippers_closed

def block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=1e-3):
    relative_position = np.subtract(gripper_position, block_position)

    return np.sum(np.subtract(relative_position, relative_grasp_position)**2) < atol

def grippers_are_closed(obs, atol=1e-3):
    gripper_state = obs['observation'][9:11]
    return abs(gripper_state[0]) - 0.024 <= atol

def grippers_are_open(obs, atol=1e-3):
    gripper_state = obs['observation'][9:11]

    return abs(gripper_state[0] - 0.05) <= atol


def pick_at_position(obs, block_position, place_position, relative_grasp_position=(0., 0., -0.02), workspace_height=0.1, atol=1e-3):
    """
    Returns
    -------
    action : [float] * 4
    """
    gripper_position = obs['observation'][:3]

    # If the gripper is already grasping the block
    if block_is_grasped(obs, gripper_position, block_position, relative_grasp_position, atol=atol):

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
        return get_move_action(obs, target_position, atol=atol, close_gripper=True)

    # If the block is ready to be grasped
    if block_inside_grippers(gripper_position, block_position, relative_grasp_position, atol=atol):

        # Close the grippers
        if DEBUG:
            print("Close the grippers")
        return np.array([0., 0., 0., -1.])

    # If the gripper is above the block
    target_position = np.add(block_position, relative_grasp_position)    
    if (gripper_position[0] - target_position[0])**2 + (gripper_position[1] - target_position[1])**2 < atol:

        # If the grippers are closed, open them
        if not grippers_are_open(obs, atol=atol):
            if DEBUG:
                print("Open the grippers")
            return np.array([0., 0., 0., 1.])

        # Move down to grasp
        if DEBUG:
            print("Move down to grasp")
        return get_move_action(obs, target_position, atol=atol)


    # Else move the gripper to above the block
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the block")
    return get_move_action(obs, target_position, atol=atol)


def get_hook_control(obs, atol=1e-2):
    """
    Returns
    -------
    action : [float] * 4
    """
    gripper_position = obs['observation'][:3]
    block_position = obs['observation'][3:6]
    hook_position = obs['observation'][25:28]
    place_position = obs['desired_goal']

    # Done
    if abs(block_position[0] - place_position[0]) + abs(block_position[1] - place_position[1]) <= 1e-2:
        if DEBUG:
            print("DONE")
        return np.array([0., 0., 0., -1.])

    # Grasp and lift the hook
    if not block_is_grasped(obs, gripper_position, hook_position, relative_grasp_position=(0., 0., -0.05), atol=atol):
        if DEBUG:
            print("Grasping and lifting the hook")
        hook_target = hook_position.copy()
        hook_target[2] = 0.5
        return pick_at_position(obs, hook_position, hook_target, relative_grasp_position=(0., 0., -0.05))

    # Align the hook to sweep
    hook_target = np.array([block_position[0] - 0.5, block_position[1] - 0.05, 0.45])

    if hook_position[0] >= hook_target[0] + 0.1 or hook_position[1] + 0.1 <= hook_target[1]:
        if DEBUG:
            print("Aligning to sweep", hook_position[0] + atol, hook_target[0], hook_position[1] + atol, hook_target[1])
        return get_move_action(obs, hook_target, close_gripper=True)

    if DEBUG:
        print("Sweeping back")

    direction = np.subtract(place_position, block_position)
    direction = direction[:2] / np.linalg.norm(direction[:2])

    return np.array([0.4 * direction[0], 0.4 * direction[1], 0., -1.])







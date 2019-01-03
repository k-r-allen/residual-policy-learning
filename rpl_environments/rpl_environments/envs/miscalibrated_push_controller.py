from .pick_and_place_controller import get_move_action

import numpy as np

DEBUG = False


def get_push_control(obs, atol=1e-3, block_width=0.15, workspace_height=0.1):

    gripper_position = obs['observation'][:3]
    block_position = obs['observation'][3:6]
    goal = obs['desired_goal']

    desired_block_angle = np.arctan2(goal[0] - block_position[0], goal[1] - block_position[1])
    gripper_angle = np.arctan2(goal[0] - gripper_position[0], goal[1] - gripper_position[1])

    push_position = block_position.copy()
    push_position[0] += -1. * np.sin(desired_block_angle) * block_width / 2.
    push_position[1] += -1. * np.cos(desired_block_angle) * block_width / 2.
    push_position[2] += 0.005

    # If the block is already at the place position, do nothing
    if np.sum(np.subtract(block_position, goal)**2) < atol:
        if DEBUG:
            print("The block is already at the place position; do nothing")
        return np.array([0., 0., 0., 0.])

    # Angle between gripper and goal vs block and goal is roughly the same
    angle_diff = abs((desired_block_angle - gripper_angle + np.pi) % (2*np.pi) - np.pi)

    gripper_sq_distance = (gripper_position[0] - goal[0])**2 + (gripper_position[1] - goal[1])**2
    block_sq_distance = (block_position[0] - goal[0])**2 + (block_position[1] - goal[1])**2

    if (gripper_position[2] - push_position[2])**2 < atol and angle_diff < np.pi/4 and block_sq_distance < gripper_sq_distance:

        # Push towards the goal
        target_position = goal
        target_position[2] = gripper_position[2]
        if DEBUG:
            print("Push")
        return get_move_action(obs, target_position, atol=atol, gain=5.0)

    # If the gripper is above the push position
    if (gripper_position[0] - push_position[0])**2 + (gripper_position[1] - push_position[1])**2 < atol:

        # Move down to prepare for push
        if DEBUG:
            print("Move down to prepare for push")
        return get_move_action(obs, push_position, atol=atol)


    # Else move the gripper to above the push
    target_position = push_position.copy()
    target_position[2] += workspace_height
    if DEBUG:
        print("Move to above the push position")

    return get_move_action(obs, target_position, atol=atol)



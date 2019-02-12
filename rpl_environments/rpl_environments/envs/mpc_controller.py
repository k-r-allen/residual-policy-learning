import copy
import numpy as np
import scipy
import heapq as hq



class MPCController(object):
    def __init__(self, model, action_scale, num_expansions=10):
        self.model = model
        self.model.reset()
        self.action_scale = action_scale
        self.num_expansions = num_expansions

    def act(self, obs, debug_info):
        queue = []

        initial_cost = self.heuristic(obs, [])
        root = initial_cost + ((obs, debug_info, []),)
        hq.heappush(queue, root)

        visited = set()

        best_node = root
        best_node_cost = initial_cost

        for _ in range(self.num_expansions):
            obs, debug_info, actions_from_root = hq.heappop(queue)[-1]

            for action in self.actions:

                child_actions_from_root = actions_from_root + [action]

                if tuple(np.sum(child_actions_from_root, axis=0)) in visited:
                    continue

                self.model.env.goal = obs['desired_goal']
                mj_sim_state = self.model.env.sim.get_state()
                mj_sim_state.qpos[:] = debug_info['sim_state']['qpos'][:]
                mj_sim_state.qvel[:] = debug_info['sim_state']['qvel'][:]
                self.model.env.sim.set_state(mj_sim_state)
                self.model.env.sim.forward()

                child_obs, _, _, child_debug_info = self.model.step(self.action_scale * np.array(action))
                mj_sim_state = self.model.env.sim.get_state()
                qpos = mj_sim_state.qpos.copy()
                qvel = mj_sim_state.qvel.copy()
                sim_state = {'qpos' : qpos, 'qvel' : qvel}
                child_debug_info['sim_state'] = sim_state

                cost = self.heuristic(child_obs, child_actions_from_root)

                node = cost + ((child_obs, child_debug_info, child_actions_from_root),)

                if cost[0] < best_node_cost[0]:
                    best_node = node
                    best_node_cost = cost
                elif cost[0] == best_node_cost[0] and cost[1] <= best_node_cost[1]:
                    best_node = node
                    best_node_cost = cost

                hq.heappush(queue, node)

                visited.add(tuple(np.sum(child_actions_from_root, axis=0)))

        best_actions = best_node[-1][-1]
        if len(best_actions) == 0:
            return np.zeros(4)

        return best_actions[0]

    def seed(self, seed):
        self.model.seed(seed=seed)

    @property
    def actions(self):
        scale = 1.

        return [ 
            (-scale, 0., 0., 0.),
            (scale, 0., 0., 0.),
            (0., -scale, 0., 0.),
            (0., scale, 0., 0.),
            (0., 0., -scale, 0.),
            (-0., 0., scale, 0.),
        ]

    def heuristic(self, obs, actions_from_root):
        gripper_position = obs['observation'][:3]
        block_position = obs['observation'][3:6]
        goal = obs['desired_goal']

        block_width = 0.1
        block_to_goal_angle = np.arctan2(goal[0] - block_position[0], goal[1] - block_position[1])
        target_gripper_position = block_position.copy()
        target_gripper_position[0] += -1. * np.sin(block_to_goal_angle) * block_width / 2.
        target_gripper_position[1] += -1. * np.cos(block_to_goal_angle) * block_width / 2.
        target_gripper_position[2] += 0.005

        object_to_goal_distance = int(10000 * np.linalg.norm(np.subtract(block_position[:2], goal[:2])))
        gripper_to_target_distance = len(actions_from_root) + int(10000 * np.linalg.norm(np.subtract(gripper_position, target_gripper_position)))

        return (object_to_goal_distance, gripper_to_target_distance, np.random.random())

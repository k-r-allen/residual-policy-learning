import numpy as np


class Controller(object):

    def reset(self, observation):
        self.last_observation = observation
        print('resetting',self.last_observation)

    def observe(self, observation, reward):
        self.last_observation = observation

    def get_action(self):
        """
        Returns
        -------
        action : Any
        done : bool
        """
        raise NotImplementedError()



class StatelessController(Controller):

    def __init__(self, controller_fn):
        self.controller_fn = controller_fn

    def get_action(self):
        return self.controller_fn(self.last_observation)



class StateMachineController(Controller):
    def __init__(self, subcontrollers, is_stalled=None, state_timeout=2, repeat=False):
        self.subcontrollers = subcontrollers
        self.is_stalled = is_stalled
        self.state_timeout = state_timeout
        self.state = 0
        self.timeout_counter = 0
        self.repeat = repeat

    def reset(self, observation):
        self.state = 0
        self.timeout_counter = 0
        self.subcontrollers[0].reset(observation)
        self.last_observation = observation

    def observe(self, observation, reward):
        if self.is_stalled and self.is_stalled():
            self.timeout_counter += 1

        self.last_observation = observation
        print('observing:',observation)
        self.subcontrollers[self.state].observe(observation, reward)

    def get_action(self):
        subcontroller = self.subcontrollers[self.state]

        action, subcontroller_done = subcontroller.get_action()

        if self.timeout_counter >= self.state_timeout:
            self.timeout_counter = 0
            subcontroller_done = True

        done = False
        if subcontroller_done:
            if self.state < len(self.subcontrollers) - 1:
                self.state += 1
                self.subcontrollers[self.state].reset(self.last_observation)
            elif self.repeat:
                self.reset(self.last_observation)
            else:
                done = True

        return action, done




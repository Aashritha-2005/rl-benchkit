class BaseAgent:
    def act(self, state):
        raise NotImplementedError

    def learn(self):
        raise NotImplementedError

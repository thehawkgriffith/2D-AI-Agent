# 2-Dimensional Artificial Intelligence Reinforcement Learning Agent

# State class for a state object
class State():
    
    def __init__(self, value, default = False, reward = 0):
        self.sprimes = [] # [Up, Down, Left, Right] Next States
        self.value = value # Value of the State 
        self.terminal = default # Is it a terminal State or not
        self.reward = reward # Reward you get by arriving to this State

# Function to print the map
def displayMap(state):
    Maprow1 = np.array(['-', '-', '-', '$'])
    Maprow2 = np.array(['-', '#', '-', 'X'])
    Maprow3 = np.array(['-', '-', '-', '-'])
    if(state == s1):
        Maprow1[0] = 'o'
    if(state == s2):
        Maprow1[1] = 'o'
    if(state == s3):
        Maprow1[2] = 'o'
    if(state == s4):
        Maprow1[3] = 'o'
    if(state == s5):
        Maprow2[0] = 'o'
    if(state == s7):
        Maprow2[2] = 'o'
    if(state == s8):
        Maprow2[3] = 'o'
    if(state == s9):
        Maprow3[0] = 'o'
    if(state == s10):
        Maprow3[1] = 'o'
    if(state == s11):
        Maprow3[2] = 'o'
    if(state == s12):
        Maprow3[3] = 'o'
    print("\nEnvironment now is:")
    print(Maprow1)
    print(Maprow2)
    print(Maprow3)

# Check if terminal state has been reached
def checkTermination(state):
    if (state == s4 or state == s8):
        return True


# Initialisation of Environment
s1 = State(0.5)
s2 = State(0.5)
s3 = State(0.5)
s4 = State(0.5, True, 1)
s5 = State(0.5)
s6 = State(0.5)
s7 = State(0.5)
s8 = State(0.5, True, -1)
s9 = State(0.5)
s10 = State(0.5)
s11 = State(0.5)
s12 = State(0.5)

s1.sprimes = [s1, s5, s1, s2]
s2.sprimes = [s2, s2, s1, s3]
s3.sprimes = [s3, s7, s2, s4]
s5.sprimes = [s1, s9, s5, s5]
s7.sprimes = [s3, s11, s7, s8]
s9.sprimes = [s5, s9, s9, s10]
s10.sprimes = [s10, s10, s9, s11]
s11.sprimes = [s7, s11, s10, s12]
s12.sprimes = [s8, s12, s11, s12]
# End of initialisation of Environment

# Agent Class
class Agent:
    
    def __init__(self):
        self.history = [] #To store all previous states to calculate Value function
        self.alpha = 0.1
    
    def takeAction(self, state, train = True):
        displayMap(state)
        eps = 0.1 #Epsilon

        #Epsilon-Greedy Strategy
        if eps < np.random.random():
            k = 0
            for prime in state.sprimes:
                if prime.value > k:
                    beststate = prime
                    k = prime.value
            nextstate = beststate
        else:
            temp = np.random.choice([0, 1, 2, 3])
            nextstate = state.sprimes[temp]
        self.history.append(nextstate)
        if checkTermination(nextstate):
            displayMap(nextstate)
            self.update(nextstate)
        else:
            self.takeAction(nextstate)
    
    def resetHistory(self):
        self.history = []
    
    def update(self, state):
        reward = state.reward
        for prev in reversed(self.history):
            val = prev.value + self.alpha*(reward - prev.value)
            prev.value = val
            reward = val
        self.resetHistory()
        
agent = Agent()

# Training with 100 epochs
for i in range(100):
    agent.takeAction(np.random.choice([s9, s12, s10, s5, s1]))

# Demonstration with any random initial state
agent.takeAction(np.random.choice([s9, s12, s10, s5, s1]))


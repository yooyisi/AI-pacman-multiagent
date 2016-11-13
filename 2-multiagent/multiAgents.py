# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        #print "ghost state: " + str(newGhostStates)
        #print newFood
        #print newPos

        distance_closest_ghost = 999999
        distance_2nd_closest_ghost = 999999
        num_ghost = 0
        for ghost in newGhostStates:
            num_ghost += 1
            md = manhattanDistance(newPos,ghost.getPosition())
            if num_ghost==1:
                distance_closest_ghost = md
            elif md < distance_closest_ghost:
                distance_closest_ghost = md
            else:
                distance_2nd_closest_ghost = md
        #print distance_closest_ghost
        #print successorGameState.getScore()
        """
        if num_ghost==2:
            if distance_2nd_closest_ghost==distance_closest_ghost==1:
                return -2
        """
        # the bigger the better
        if distance_closest_ghost==0:
            return -9999
        if distance_closest_ghost==1:
            return -999
        #print "score "+str(successorGameState.getScore())+" ghost "+str(distance_closest_ghost)
        #print successorGameState.getScore()-distance_closest_ghost*10
        return successorGameState.getScore()-distance_closest_ghost*10

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        num_agent = gameState.getNumAgents()
        depth = self.depth
        num_steps = depth*num_agent#(depth+1)%2 + (depth+1)*num_agent/2
        actions = gameState.getLegalActions(0)
        best_choice = self.minimax_function(depth, 1, num_agent, gameState, num_steps)
        return actions[best_choice]

    # current_depth starts from 1 ends at depth
    def minimax_function(self, depth, current_step, num_agent, gameState, num_steps):
        if current_step>num_steps or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        agent_index = current_step % num_agent - 1
        legal_moves = gameState.getLegalActions(agent_index)
        values = [self.minimax_function(depth, current_step + 1, num_agent, gameState.generateSuccessor(agent_index, a), num_steps) for a in legal_moves]

        if (current_step -1)%num_agent == 0:
            values += [-float("inf")]
            max_value = max(values)
            # for 1st, return the action
            if current_step == 1:
                best_indices = [index for index in range(len(values)-1) if values[index] == max_value]
                return random.choice(best_indices) # Pick randomly among the best
            # for others, return the max value
            return max_value
        else:
            values += [float("inf")]
            return min(values)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()
        depth = self.depth
        num_steps = depth * num_agent - 1  # start from 0
        actions = gameState.getLegalActions(0)
        best_choice = self.pruning_function(depth, 0, num_agent, gameState, num_steps, -float("inf"), float("inf") )
        return actions[best_choice]

    # current_depth starts from 1 ends at depth
    def pruning_function(self, depth, current_step, num_agent, gameState, num_steps, alpha, beta):
        if current_step>num_steps or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        agent_index = current_step % num_agent
        isPacman = agent_index == 0
        legal_moves = gameState.getLegalActions(agent_index)

        # initialize v
        if not isPacman:
            v = float("inf")
        if isPacman:
            v = -float("inf")

        # for each node, check alpha-beta condition and expand the node
        values = [v]
        for a in legal_moves:
            if isPacman and (v>beta):
                break
            if (not isPacman) and (v<alpha):
                break
            values += [self.pruning_function(depth,current_step+1, num_agent, gameState.generateSuccessor(agent_index, a), num_steps, alpha, beta)]
            # update v , alpha and beta
            if isPacman:
                v = max(values)
                alpha = max(v,alpha)
            if not isPacman:
                v = min(values)
                beta = min(v, beta)

        if current_step == 0:
            best_indices = [index - 1 for index in range(len(values)) if values[index] == v]
            return random.choice(best_indices)  # Pick randomly among the best

        return v

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        num_agent = gameState.getNumAgents()
        depth = self.depth
        num_steps = depth * num_agent - 1 # (depth+1)%2 + (depth+1)*num_agent/2
        actions = gameState.getLegalActions(0)
        best_choice = self.expectimax_function(depth, 0, num_agent, gameState, num_steps)
        return actions[best_choice]

    # current_depth starts from 1 ends at depth
    def expectimax_function(self, depth, current_step, num_agent, gameState, num_steps):
        if current_step>num_steps or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        agent_index = current_step % num_agent
        isPacman = agent_index == 0
        legal_moves = gameState.getLegalActions(agent_index)
        values = [self.expectimax_function(depth, current_step + 1, num_agent, gameState.generateSuccessor(agent_index, a), num_steps) for a in legal_moves]

        if isPacman:
            values += [-float("inf")]
            v = max(values)
            # for 1st, return the action
            if current_step == 0:
                best_indices = [index for index in range(len(values)-1) if values[index] == v]
                return random.choice(best_indices) # Pick randomly among the best
            # for others, return the max value
            return v
        else:
            sumGhost = 0
            for value in values:
                sumGhost += value
            return float(sumGhost)/float(len(values))

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


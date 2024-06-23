from util import manhattanDistance
from game import Directions
import random, util
from game import Agent
from math import dist

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            elif agentIndex == 0:  # Pacman's turn (maximizing)
                maxEval = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.getNextState(agentIndex, action)
                    maxEval = max(maxEval, minimax(successor, depth, 1))
                return maxEval
            else: # Ghost's turn (minimizing)
                minEval = float("inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.getNextState(agentIndex, action)
                    if agentIndex == state.getNumAgents() - 1:
                        minEval = min(minEval, minimax(successor, depth - 1, 0))
                    else:
                        minEval = min(minEval, minimax(successor, depth, agentIndex + 1))
                return minEval

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.getNextState(0, action)
            value  = minimax(successor, self.depth, 1) # Start with depth 1 for ghosts
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        def alphaBeta(state, depth, alpha, beta, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            elif agentIndex==0: # Pacman's turn (maximizing)
                return max_value(state, depth, alpha, beta, agentIndex)
            else:  # Ghost's turn (minimizing)
                return min_value(state, depth, alpha, beta, agentIndex)
        def max_value(state, depth, alpha, beta, agentIndex):
            maxEval = float("-inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.getNextState(agentIndex, action)
                eval = alphaBeta(successor, depth, alpha, beta, 1)
                maxEval = max(maxEval, eval)
                if beta < maxEval:
                    return maxEval
                alpha = max(alpha, maxEval)
            return maxEval
        
        def min_value(state, depth, alpha, beta, agentIndex):
            minEval = float("inf")
            for action in state.getLegalActions(agentIndex):
                successor = state.getNextState(agentIndex, action)
                if agentIndex == state.getNumAgents()-1:
                    eval = alphaBeta(successor, depth-1, alpha, beta, 0)
                    minEval = min(minEval, eval)
                else:
                    eval = alphaBeta(successor, depth, alpha, beta, agentIndex+1)
                    minEval = min(minEval, eval)
                if minEval < alpha:
                    return minEval
                beta = min(beta, minEval)
            return minEval

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for action in legalActions:
            successor = gameState.getNextState(0, action)
            value  = alphaBeta(successor, self.depth, alpha, beta, 1) # Start with depth 1 for ghosts
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue) 
        return bestAction
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        def expectimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            elif agentIndex == 0:  # Pacman's turn (maximizing)
                maxEval = float("-inf")
                for action in state.getLegalActions(agentIndex):
                    successor = state.getNextState(agentIndex, action)
                    maxEval = max(maxEval, expectimax(successor, depth, 1))
                return maxEval
            else: # Ghost's turn (minimizing)
                expEval = 0
                actions = state.getLegalActions(agentIndex)
                probability = 1.0 / len(actions)
                for action in actions:
                    successor = state.getNextState(agentIndex, action)
                    if agentIndex == state.getNumAgents() - 1:
                        expEval += probability * expectimax(successor, depth-1, 0)
                    else:
                        expEval += probability * expectimax(successor, depth, agentIndex+1)
                return expEval

        legalActions = gameState.getLegalActions(0)
        bestAction = None
        bestValue = float("-inf")
        for action in legalActions:
            successor = gameState.getNextState(0, action)
            value  = expectimax(successor, self.depth, 1) # Start with depth 1 for ghosts
            if value > bestValue:
                bestValue = value
                bestAction = action
        return bestAction
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    # initialize the basic information
    pacman_position = currentGameState.getPacmanPosition()
    ghost_positions = currentGameState.getGhostPositions()
    score = currentGameState.getScore()
    capsule_list = currentGameState.getCapsules()
    capsule_count = len(capsule_list)
    food_list = currentGameState.getFood().asList()
    food_count = len(food_list)
    nearest_food = 1
    nearest_ghost = 1
    nearest_capsule = 1

    # find the distance from pacman to each food
    food_distance = [manhattanDistance(pacman_position, food_position) for food_position in food_list]
    if food_count > 0:
        nearest_food = min(food_distance)

    # find the distance from pacman to each capsule
    capsule_distance = [manhattanDistance(pacman_position, capsule_position) for capsule_position in capsule_list]
    if capsule_count > 0:
        nearest_capsule = min(capsule_distance)
    #print(nearest_capsule)
    #print(type(nearest_capsule))

    # find the distance from pacman to each ghost
    ghost_distance = [manhattanDistance(pacman_position, ghost_position) for ghost_position in ghost_positions]
    if len(ghost_distance) > 0:
        nearest_ghost = min(ghost_distance)
        
    
    # If pacman is too close to the ghost,
    # then let it escape first by setting
    # the distance to the nearest food
    if nearest_ghost <= 1 or nearest_ghost > 13: 
        nearest_food = 10000000
    if nearest_ghost==0:
        nearest_ghost = 0.0001

    features = [1./nearest_food, score, food_count, capsule_count, 1./nearest_capsule, 1./nearest_ghost]

    weights = [50, 200, -100, -150, 60, 5]

    # Linear combination of features
    return sum([feature * weight for feature, weight in zip(features, weights)])
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction

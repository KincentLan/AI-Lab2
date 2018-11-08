# AI Lab 2: Games and ConnectFour 

# Name(s): 
# Email(s): 

from game_api import *
from boards import *
from toytree import GAME1
from time import time

INF = float('inf')


# Please see wiki lab page for full description of functions and API.

#### Part 1: Utility Functions #################################################

def is_game_over_connectfour(board):
    """Returns True if game is over, otherwise False."""
    if board.count_pieces(True) + board.count_pieces(False) == board.num_rows * board.num_cols:
        return True

    chains = board.get_all_chains()

    for chain in chains:
        if len(chain) >= 4:
            return True

    return False


def next_boards_connectfour(board):
    """Returns a list of ConnectFourBoard objects that could result from the
    next move, or an empty list if no moves can be made."""
    if is_game_over_connectfour(board):
        return []
    next_boards = []
    for col in range(0, board.num_cols):
        if not board.is_column_full(col):
            next_boards.append(board.add_piece(col))

    return next_boards


def endgame_score_connectfour(board, is_current_player_maximizer):
    """Given an endgame board, returns 1000 if the maximizer has won,
    -1000 if the minimizer has won, or 0 in case of a tie."""
    chains = board.get_all_chains(is_current_player_maximizer)
    for chain in chains:
        if len(chain) >= 4:
            return 1000

    chains = board.get_all_chains(not is_current_player_maximizer)
    for chain in chains:
        if len(chain) >= 4:
            return -1000

    return 0


def endgame_score_connectfour_faster(board, is_current_player_maximizer):
    """Given an endgame board, returns an endgame score with abs(score) >= 1000,
    returning larger absolute scores for winning sooner."""
    if not is_current_player_maximizer:
        return 1000 + 42 - board.count_pieces()
    else:
        return -1 * (1000 + 42 - board.count_pieces())


# Now we can create AbstractGameState objects for Connect Four, using some of
# the functions you implemented above.  You can use the following examples to
# test your dfs and minimax implementations in Part 2.

# This AbstractGameState represents a new ConnectFourBoard, before the game has started:
state_starting_connectfour = AbstractGameState(snapshot=ConnectFourBoard(),
                                               is_game_over_fn=is_game_over_connectfour,
                                               generate_next_states_fn=next_boards_connectfour,
                                               endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "NEARLY_OVER" from boards.py:
state_NEARLY_OVER = AbstractGameState(snapshot=NEARLY_OVER,
                                      is_game_over_fn=is_game_over_connectfour,
                                      generate_next_states_fn=next_boards_connectfour,
                                      endgame_score_fn=endgame_score_connectfour_faster)

# This AbstractGameState represents the ConnectFourBoard "BOARD_UHOH" from boards.py:
state_UHOH = AbstractGameState(snapshot=BOARD_UHOH,
                               is_game_over_fn=is_game_over_connectfour,
                               generate_next_states_fn=next_boards_connectfour,
                               endgame_score_fn=endgame_score_connectfour_faster)


#### Part 2: Searching a Game Tree #############################################

# Note: Functions in Part 2 use the AbstractGameState API, not ConnectFourBoard.

def dfs_maximizing(state):
    """Performs depth-first search to find path with highest endgame score.
    Returns a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    bestRetTup = [[state], 0, 0]
    if state.is_game_over():
        return tuple([[state], state.get_endgame_score(True), 1])

    for neighbor in state.generate_next_states():
        smallerRetTup = dfs_maximizing(neighbor)
        bestRetTup[2] = bestRetTup[2] + smallerRetTup[2]

        if bestRetTup[1] < smallerRetTup[1]:
            bestRetTup[0] = [state]
            for x in smallerRetTup[0]:
                bestRetTup[0].append(x)
            bestRetTup[1] = smallerRetTup[1]
    return tuple(bestRetTup)


# Uncomment the line below to try your dfs_maximizing on an
# AbstractGameState representing the games tree "GAME1" from toytree.py:

#pretty_print_dfs_type(dfs_maximizing(GAME1))


def minimax_endgame_search(state, maximize=True):
    """Performs minimax search, searching all leaf nodes and statically
    evaluating all endgame scores.  Returns the same as dfs_maximizing:
    a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    if state.is_game_over():
        return tuple([[state], state.get_endgame_score(maximize), 1])

    if maximize:
        maxEval = [[], -INF, 0]
        for neighbor in state.generate_next_states():
            eval = minimax_endgame_search(neighbor, not maximize)
            maxEval[2] = maxEval[2] + eval[2]

            if maxEval[1] < eval[1]:
                maxEval[0] = [state]
                for x in eval[0]:
                    maxEval[0].append(x)
                maxEval[1] = eval[1]

        return maxEval

    minEval = [[], INF, 0]
    for neighbor in state.generate_next_states():
        eval = minimax_endgame_search(neighbor, not maximize)
        minEval[2] = minEval[2] + eval[2]

        if minEval[1] > eval[1]:
            minEval[0] = [state]
            for x in eval[0]:
                minEval[0].append(x)
            minEval[1] = eval[1]

    return minEval


# Uncomment the line below to try your minimax_endgame_search on an
# AbstractGameState representing the ConnectFourBoard "NEARLY_OVER" from boards.py:

# pretty_print_dfs_type(minimax_endgame_search(state_NEARLY_OVER))


#### Part 3: Cutting off and Pruning search #############################################


def heuristic_connectfour(board, is_current_player_maximizer):
    """Given a non-endgame board, returns a heuristic score with
    abs(score) < 1000, where higher numbers indicate that the board is better
    for the maximizer."""
    heuristic = 0
    for chain in board.get_all_chains(is_current_player_maximizer):
        heuristic += len(chain) * len(chain)

    for chain in board.get_all_chains(not is_current_player_maximizer):
        heuristic -= len(chain) * len(chain)

    return heuristic


## Note that the signature of heuristic_fn is heuristic_fn(board, maximize=True)

def minimax_search(state, heuristic_fn=always_zero, depth_limit=INF, maximize=True):
    """Performs h-minimax, cutting off search at depth_limit and using heuristic_fn
    to evaluate non-terminal states.
    Same return type as dfs_maximizing, a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    if state.is_game_over():
        return tuple([[state], state.get_endgame_score(maximize), 1])

    if depth_limit == 0:
        return tuple([[state], heuristic_fn(state.get_snapshot(), maximize), 1])

    neighbors = []
    for neighbor in state.generate_next_states():
        neighbors.append((heuristic_fn(neighbor.get_snapshot(), maximize), neighbor))

    sorted(neighbors, key=lambda x: x[0])

    if maximize:
        maxEval = [[], -INF, 0]
        for neighbor in neighbors:
            eval = minimax_search(neighbor[1], heuristic_fn, depth_limit-1, not maximize)
            maxEval[2] = maxEval[2] + eval[2]

            if maxEval[1] < eval[1]:
                maxEval[0] = [state]
                for x in eval[0]:
                    maxEval[0].append(x)
                maxEval[1] = eval[1]

        return maxEval

    minEval = [[], INF, 0]
    for neighbor in neighbors:
        eval = minimax_search(neighbor[1], heuristic_fn, depth_limit-1, not maximize)
        minEval[2] = minEval[2] + eval[2]

        if minEval[1] > eval[1]:
            minEval[0] = [state]
            for x in eval[0]:
                minEval[0].append(x)
            minEval[1] = eval[1]

    return minEval



# Uncomment the line below to try minimax_search with "BOARD_UHOH" and
# depth_limit=1. Try increasing the value of depth_limit to see what happens:

#pretty_print_dfs_type(minimax_search(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=2))

def minimax_search_alphabeta(state, alpha=-INF, beta=INF, heuristic_fn=always_zero,
                             depth_limit=INF, maximize=True):
    """"Performs minimax with alpha-beta pruning.
    Same return type as dfs_maximizing, a tuple containing:
     0. the best path (a list of AbstractGameState objects),
     1. the score of the leaf node (a number), and
     2. the number of static evaluations performed (a number)"""
    if state.is_game_over():
        return tuple([[state], state.get_endgame_score(maximize), 1])

    if depth_limit == 0:
        return tuple([[state], heuristic_fn(state.get_snapshot(), maximize), 1])


    neighbors = []
    for neighbor in state.generate_next_states():
        if heuristic_fn(neighbor.get_snapshot(), maximize) != None:
            neighbors.append((heuristic_fn(neighbor.get_snapshot(), maximize), neighbor))
        else:
            neighbors.append((0, neighbor))

    sorted(neighbors, key=lambda x: x[0])

    if maximize:
        maxEval = [[], -INF, 0]
        for neighbor in neighbors:
            eval = minimax_search_alphabeta(neighbor[1], alpha, beta, heuristic_fn, depth_limit-1, not maximize)
            maxEval[2] = maxEval[2] + eval[2]

            if maxEval[1] < eval[1]:
                maxEval[0] = [state]
                for x in eval[0]:
                    maxEval[0].append(x)
                maxEval[1] = eval[1]

            alpha = max(alpha, eval[1])
            if beta <= alpha:
                break

        return maxEval

    minEval = [[], INF, 0]
    for neighbor in neighbors:
        eval = minimax_search_alphabeta(neighbor[1], alpha, beta, heuristic_fn, depth_limit-1, not maximize)
        minEval[2] = minEval[2] + eval[2]

        if minEval[1] > eval[1]:
            minEval[0] = [state]
            for x in eval[0]:
                minEval[0].append(x)
            minEval[1] = eval[1]

        beta = min(beta, eval[1])
        if beta <= alpha:
            break

    return minEval


# Uncomment the line below to try minimax_search_alphabeta with "BOARD_UHOH" and
# depth_limit=4. Compare with the number of evaluations from minimax_search for
# different values of depth_limit.

# pretty_print_dfs_type(minimax_search_alphabeta(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4))


def progressive_deepening(state, heuristic_fn=always_zero, depth_limit=INF,
                          maximize=True, time_limit=INF):
    """Runs minimax with alpha-beta pruning. At each level, updates anytime_value
    with the tuple returned from minimax_search_alphabeta.
    Returns anytime_value."""


# Uncomment the line below to try progressive_deepening with "BOARD_UHOH" and
# depth_limit=4. Compare the total number of evaluations with the number of
# evaluations from minimax_search or minimax_search_alphabeta.

# progressive_deepening(state_UHOH, heuristic_fn=heuristic_connectfour, depth_limit=4).pretty_print()


# Progressive deepening is NOT optional. However, you may find that
#  the tests for progressive deepening take a long time. If you would
#  like to temporarily bypass them, set this variable False. You will,
#  of course, need to set this back to True to pass all of the local
#  and online tests.
TEST_PROGRESSIVE_DEEPENING = True
if not TEST_PROGRESSIVE_DEEPENING:
    def not_implemented(*args): raise NotImplementedError
    progressive_deepening = not_implemented


#
# If you want to enter the tournament, implement your final contestant
# in this function. You may write other helper functions/classes
# but the function must take these arguments (though it can certainly ignore them)
# and must return an AnytimeValue.
#
def tournament_search(state, heuristic_fn=always_zero, depth_limit=INF,
                      maximize=True, time_limit=INF):
    """Runs some kind of search (probably progressive deepening).
    Returns an AnytimeValue."""
    raise NotImplementedError
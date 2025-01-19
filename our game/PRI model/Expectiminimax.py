import numpy as np
import copy
import multiprocessing as mp

INF = 2 ** 64

# Directions your Game2048 supports.
# We'll call game.move_left(), game.move_right(), etc. based on these.
directions = ['left', 'right', 'up', 'down']

# A sample "snake" weighting pattern, just like in your old code
PERFECT_SNAKE = [
    [2, 4, 8, 16],
    [256, 128, 64, 32],
    [512, 1024, 2048, 4096],
    [65536, 32768, 16384, 8192]
]


def snakeHeuristic(game):
    """
    Example heuristic that multiplies each tile by a 'snake' pattern.
    'game' is an instance of your Game2048 class.
    We'll use game.get_board() to get the numpy array of shape (4,4).
    """
    board = game.get_board()
    h = 0
    for i in range(4):
        for j in range(4):
            tile_value = board[i, j]
            h += tile_value * PERFECT_SNAKE[i][j]
    return h


def checkLoss(game):
    """
    Returns True if no moves are possible.
    In your Game2048, that means not game.can_move().
    """
    return not game.can_move()


def makeMove(game, dir):
    """
    Executes one move on a copy of the game.
    Returns True if this move caused any change, False otherwise.
    We detect 'change' by comparing the board before & after.
    """
    prevBoard = game.get_board().copy()
    if dir == 'left':
        game.move_left()
    elif dir == 'right':
        game.move_right()
    elif dir == 'up':
        game.move_up()
    elif dir == 'down':
        game.move_down()
    return not np.array_equal(prevBoard, game.get_board())


def getOpenTiles(game):
    """
    Returns a list of the (row, col) positions that are empty (== 0).
    We'll use this in 'Nature's turn' to place possible new tiles.
    """
    board = game.get_board()
    empty_positions = np.argwhere(board == 0)
    return empty_positions


def expectiminimax(game, depth, lastDir=None):
    """
    The recursive expectiminimax function.

    - If there are no moves left, treat it as a losing position => return (-INF).
    - If depth < 0, we've reached the cutoff => return the heuristic.
    - If depth is fractional, it's the AI's turn (we pick the best move).
    - If depth is an integer, it's 'Nature's turn' (we average over new-tile placements).
    """
    # If game is lost, return the losing utility
    if checkLoss(game):
        return -INF, lastDir

    # If we've reached our search cutoff, evaluate the board
    if depth < 0:
        return snakeHeuristic(game), lastDir

    # Player's turn (depth is .5, 1.5, 2.5, ...)
    if depth != int(depth):
        bestVal = -INF
        bestDirection = lastDir
        # Try all possible moves
        for d in directions:
            simGame = copy.deepcopy(game)
            changed = makeMove(simGame, d)
            if changed:
                val, _ = expectiminimax(simGame, depth - 0.5, d)
                if val > bestVal:
                    bestVal = val
                    bestDirection = d
        return bestVal, bestDirection

    # Nature's turn (depth is an integer)
    else:
        openPositions = getOpenTiles(game)
        if len(openPositions) == 0:
            # No empty spots: either lost or a forced heuristic
            if checkLoss(game):
                return -INF, lastDir
            else:
                # If it’s not immediately lost, just return the heuristic
                return snakeHeuristic(game), lastDir

        # Average over all possible new-tile placements
        totalVal = 0.0
        for (r, c) in openPositions:
            # Try placing '2'
            oldVal = game.board[r, c]
            game.board[r, c] = 2
            val2, _ = expectiminimax(game, depth - 0.5, lastDir)

            # Try placing '4'
            #game.board[r, c] = 4
            #val4, _ = expectiminimax(game, depth - 0.5, lastDir)

            # Restore the old cell
            game.board[r, c] = oldVal

            # Weighted average: 2 appears 90%, 4 appears 10%
            totalVal += (val2 ) / len(openPositions)

        return totalVal, lastDir


def getNextBestMoveExpectiminimax(game, pool, depth=2):
    """
    The toplevel function that chooses the best next move
    at the start of the search. It spawns async calls for each
    possible move, then picks the best result.

    - 'game': your Game2048 instance
    - 'pool': a multiprocessing Pool object
    - 'depth': how far we search (integer => how many 'nature' layers)
    """
    bestScore = -INF
    bestMove = 'left'  # fallback if all moves are invalid

    # We'll gather results of each valid move in parallel
    asyncResults = []
    for d in directions:
        simGame = copy.deepcopy(game)
        changed = makeMove(simGame, d)
        if not changed:
            # This direction didn't move anything; skip
            continue
        # Start an async job for expectiminimax
        asyncResults.append(pool.apply_async(expectiminimax, (simGame, depth, d)))

    # If no valid moves, return None or any sentinel you prefer
    if not asyncResults:
        return None

    # Collect & find best
    results = [r.get() for r in asyncResults]  # each result is (score, direction)
    for (score, direction) in results:
        if score >= bestScore:
            bestScore = score
            bestMove = direction

    return bestMove

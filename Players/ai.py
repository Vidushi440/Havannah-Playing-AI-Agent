import time
import math
import random
import numpy as np
from collections import defaultdict
from helper import *

class MCTSNode:
    def __init__(self, state: np.array, parent=None, move=None):
        self.state = state  # The current board state at this node
        self.parent = parent  # The parent node
        self.move = move  # The move that led to this node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times this node has been visited
        self.wins = 0  # Number of wins from this node
        self.untried_moves = get_valid_actions(state)  # Moves that haven't been tried yet

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def best_child(self, exploration_weight=1.41):
        
        choices_weights = [
            (child.wins / child.visits) + exploration_weight * math.sqrt(2 * math.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(choices_weights)]


class MCTS:
    def __init__(self, exploration_weight=1.41):
        self.exploration_weight = exploration_weight
        self.num_rollouts = 0
        self.run_time = 0

    def search(self, root: MCTSNode, player: int, opponent: int, time_limit: float):
        start_time = time.process_time()

        while time.process_time() - start_time < time_limit:
            node = self._select(root)
            if not node.is_fully_expanded():
                node = self._expand(node, player)
            outcome = self._simulate(node, player, opponent)
            self._backpropagate(node, outcome, player)

            self.num_rollouts += 1  

        self.run_time = time.process_time() - start_time  

        return root.best_child(0)  

    def manhattan_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> int:
        """
        Calculates the Manhattan distance between two points.
        point1: (x1, y1)
        point2: (x2, y2)
        Returns: Manhattan distance between point1 and point2
        """
        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])

    def evaluate_move(self, board: np.array, move: Tuple[int, int], player: int, opponent: int) -> float:
       
        new_board = board.copy()
        score = 1  # Base score for non-critical moves

        # Check for blocking opponent’s winning moves
        new_board[move[0], move[1]] = opponent
        if check_win(new_board, move, opponent)[0]:
            score += 900  

        new_board[move[0], move[1]] = player

        if check_win(new_board, move, player)[0]:
            score += 1000

        # Priority for controlling corners
        corners = get_all_corners(board.shape[0])
        corner_distances = [self.manhattan_distance(move, corner) for corner in corners]
        if move in corners:
            score += 500 
        else:
            score += 200 / min(corner_distances)  

        # Priority for controlling edges
        edges = get_all_edges(board.shape[0])
        if any(move in edge_group for edge_group in edges):
            score += 300  
        else:
            edge_distances = [self.manhattan_distance(move, edge) for edge_group in edges for edge in edge_group]
            score += 150 / min(edge_distances)  

        opponent_positions = np.argwhere(board == opponent)
        if len(opponent_positions) > 0:
            opponent_distances = [self.manhattan_distance(move, tuple(opponent_pos)) for opponent_pos in opponent_positions]
            score += 100 / min(opponent_distances) 

        return score

    def best_move(self, root: MCTSNode):
        if len(root.children) == 0:
            return None 

        # Get the move leading to the most visited node (best move)
        max_visits = max(child.visits for child in root.children)
        best_children = [child for child in root.children if child.visits == max_visits]

        best_child = random.choice(best_children)

        return best_child.move

    def _select(self, node: MCTSNode):
       
        while node.is_fully_expanded() and len(node.children) > 0:
            node = node.best_child(self.exploration_weight)
        return node

    def _expand(self, node: MCTSNode, player: int):
        # Expand by selecting one of the untried moves
        move = node.untried_moves.pop()
        new_state = node.state.copy()
        new_state[move[0], move[1]] = player
        child_node = MCTSNode(new_state, parent=node, move=move)
        node.children.append(child_node)
        return child_node

    def _simulate(self, node: MCTSNode, player: int, opponent: int):
        
        current_state = node.state.copy()
        current_player = player

        while len(get_valid_actions(current_state)) > 0:
            # Get valid moves
            valid_moves = get_valid_actions(current_state)
            
            # Sort moves based on a heuristic evaluation function
            valid_moves.sort(key=lambda move: self.evaluate_move(current_state, move, current_player, opponent), reverse=True)

            move = valid_moves[0]
            current_state[move[0], move[1]] = current_player

            current_player = opponent if current_player == player else player

            if check_win(current_state, move, current_player)[0]:
                return 1 if current_player == player else 0  

        return 0.5 

    def _backpropagate(self, node: MCTSNode, outcome: float, player: int):
        
        while node is not None:
            node.visits += 1
            if node.parent is None or node.parent.state[node.move] == player:
                node.wins += outcome  
            else:
                node.wins += (1 - outcome)  
            node = node.parent

class AIPlayer:

    def __init__(self, player_number: int, timer):
        self.player_number = player_number
        self.type = 'ai'
        self.player_string = 'Player {}: ai'.format(player_number)
        self.timer = timer
        self.state = None

    def manhattan_distance(self, point1: Tuple[int, int], point2: Tuple[int, int]) -> int:

        return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
    
    def opp_move(self, state: np.array, opp: int) -> Tuple[int, int]:

        if self.state is None:
            return None

        # Iterate over the entire board to find where the opponent's move was made
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if self.state[i, j] == 0 and state[i, j] == opp:
                    return (i, j)  # Return the coordinates of the opponent's move
    
        return None


    def get_move(self, state: np.array) -> Tuple[int, int]:
        """
        Given the current state of the board, return the next move
        """
        time_left = fetch_remaining_time(self.timer, self.player_number)
        valid = get_valid_actions(state)
        time_per_move = time_left / max(1, (len(valid)+1)//2)  # Ensure non-zero divisor
        opponent = 2 if self.player_number == 1 else 1
         # Check if the state is None
        if state is None:
            raise NotImplementedError('Whoops I don\'t know what to do')
        last_opponent_move = self.opp_move(state, opponent)
        
        dim = (state.shape[0] + 1) // 2
        if self.state is None:
            p = random.random()
            if p > 0.6 and state[dim-1,dim-1]==0:
                current_state = state.copy()
                current_state[dim-1,dim-1]=self.player_number
                self.state= current_state
                return (dim-1,dim-1)
            else:
                # Select a random empty corner
                corners = get_all_corners(state.shape[0])
                available_corners = [corner for corner in corners if state[corner[0], corner[1]] == 0]
                if len(available_corners) > 0:
                    move = random.choice(available_corners)
                    current_state = state.copy()
                    current_state[move[0], move[1]] = self.player_number
                    self.state = current_state
                    return (int(move[0]), int(move[1]))
                else:
                    raise ValueError("No empty corners available")

           

        for move in valid:
           
            state[move[0],move[1]]=self.player_number
            if check_win(state, move, self.player_number)[0]:
                state[move[0],move[1]]=0
                current_state = state.copy()
                current_state[move[0],move[1]]=self.player_number
                self.state= current_state
                return (int(move[0]),int(move[1]))
            state[move[0],move[1]]=0

        for move in valid:
            state[move[0],move[1]]=opponent
            if check_win(state, move, opponent)[0]:
                state[move[0],move[1]]=0
                current_state = state.copy()
                current_state[move[0],move[1]]=self.player_number
                self.state= current_state
                return (int(move[0]),int(move[1]))
            state[move[0],move[1]]=0
        
        for move in valid:
            state[move[0],move[1]]=self.player_number
            valid2=get_valid_actions(state,self.player_number)
            c=0
            for move2 in valid2:
                state[move2[0],move2[1]]=self.player_number
                if check_win(state,move2,self.player_number)[0]:
                    c+=1
                    state[move2[0],move2[1]]=0
                    if c==2:
                        state[move[0],move[1]]=0
                        current_state = state.copy()
                        current_state[move[0],move[1]]=self.player_number
                        self.state= current_state
                        return (int(move[0]),int(move[1]))
                state[move2[0],move2[1]]=0
            state[move[0],move[1]]=0

        for move in valid:
            state[move[0],move[1]]=opponent
            valid2=get_valid_actions(state,opponent)
            c=0
            for move2 in valid2:
                state[move2[0],move2[1]]=opponent
                if check_win(state,move2,opponent)[0]:
                    c+=1
                    state[move2[0],move2[1]]=0
                    if c==2:
                        state[move[0],move[1]]=0
                        current_state = state.copy()
                        current_state[move[0],move[1]]=self.player_number
                        self.state= current_state
                        return (int(move[0]),int(move[1]))
                state[move2[0],move2[1]]=0
            state[move[0],move[1]]=0

        if last_opponent_move is not None:
            valid_moves_with_distances = [(move, self.manhattan_distance(move, last_opponent_move)) for move in valid]
            move = min(valid_moves_with_distances, key=lambda x: x[1])[0]

            if dim<=4:
                p=0.4
            else:
                p=0.6
            
            if random.random() < p:
                current_state = state.copy()
                current_state[move[0],move[1]]=self.player_number
                self.state= current_state
                return (int(move[0]),int(move[1]))

        # Set up the root node for MCTS
        root = MCTSNode(state)
        mcts = MCTS()

        # Run MCTS for the given time per move
        best_node = mcts.search(root, self.player_number, opponent, time_per_move)
        current_state = state.copy()
        current_state[best_node.move[0],best_node.move[1]]=self.player_number
        self.state= current_state
        # Return the move that leads to the best node
        return (int(best_node.move[0]), int(best_node.move[1]))
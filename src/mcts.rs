//! Implementation of Monte Carlo Tree Search (MCTS) algorithm.
//!
//! This module provides the core MCTS logic, allowing for tree traversal,
//! node expansion, simulation, and backpropagation. It is designed to be
//! generic over game types and evaluation strategies, making it suitable
//! for various board games.

use core::f64;
use std::{cell::RefCell, rc::Rc};

use crate::{Game, GameEvaluator, Node, NodeRef};

/// A very large floating-point number used to represent infinity in score calculations.
///
/// This constant is used, for instance, to ensure unvisited nodes are prioritized
/// during selection. Using `1e300` instead of `f64::MAX` can sometimes prevent
/// potential overflow or precision issues when `INFINITY` is involved in
/// arithmetic operations.
const INFINITY : f64 = 1e300;

/// Data stored in each node of the MCTS tree.
///
/// This struct holds the essential statistics and game-specific information
/// for a node within the Monte Carlo Search Tree.
///
/// # Type Parameters
/// - `N`: The number of possible actions in the game, fixed at compile time.
struct MctsNodeData<const N: usize>{
    /// Policy probabilities for each action from this node's state.
    /// Typically obtained from a `GameEvaluator`.
    policy: [f64; N],
    /// A boolean mask indicating which actions are valid from this node's state.
    /// `true` at index `i` means action `i` is valid.
    mask: [bool; N],
    /// The cumulative sum of scores obtained from simulations that have passed through this node.
    /// This is used to calculate the average value of the node.
    score: f64,
    /// The number of times this node has been visited during simulations.
    /// Incremented during backpropagation.
    n: usize,
    /// A flag indicating if the game state represented by this node is a terminal (finished) state.
    /// If `true`, no further actions can be taken from this node.
    finish: bool
}

impl<const N: usize> MctsNodeData<N>{
    /// Creates a new `MctsNodeData` with default values.
    ///
    /// Policy is initialized to uniform probabilities, mask to false, score and visits to zero,
    /// and finish flag to false.
    ///
    /// # Returns
    /// A new `MctsNodeData` instance.
    pub fn new() -> Self{
        MctsNodeData { 
            policy: [1./N as f64; N], 
            mask: [false; N], 
            score: 0.0, 
            n: 0, 
            finish: false 
        }
    }

    /// Calculates the average value (score per visit) of this node.
    ///
    /// # Returns
    /// The average score (`score / n`). Returns `0.0` if `n` (visit count) is zero
    /// to prevent division by zero.
    #[inline]
    pub fn get_value(&self) -> f64{
        if self.n != 0 { self.score / self.n as f64 } else { 0.0 }
    }

    /// Returns the visit count (`n`) for this node.
    ///
    /// # Returns
    /// The number of times this node has been visited.
    #[inline]
    pub fn get_n(&self) -> usize{
        self.n
    }
    
    /// Gets the policy value (action probability) for a specific action index.
    ///
    /// # Parameters
    /// - `index`: The action index (0 to N-1).
    ///
    /// # Returns
    /// The policy probability for the given action.
    #[inline]
    pub fn get_policy(&self, index: usize) -> f64{
        self.policy[index]
    }

    /// Checks if an action is valid from this node's state using the mask.
    ///
    /// # Parameters
    /// - `index`: The action index to check.
    ///
    /// # Returns
    /// `true` if the action is valid, `false` otherwise.
    #[inline]
    pub fn get_mask(&self, index: usize) -> bool{
        self.mask[index]
    }

    /// Returns whether this node represents a finished game state.
    ///
    /// # Returns
    /// `true` if the game is over at this node, `false` otherwise.
    #[inline]
    pub fn is_finish(&self) -> bool{
        self.finish
    }

    /// Updates the node's statistics by incorporating a new simulation result.
    ///
    /// This method adds the `score` to the total `score` and increments the visit count `n`.
    ///
    /// # Parameters
    /// - `score`: The result of a simulation to incorporate into this node's statistics.
    #[inline]
    pub fn add_score(&mut self, score: f64){
        self.score += score;
        self.n += 1;
    }
}

/// Type alias for a `Node` containing `MctsNodeData`.
type MctsNode<const N: usize> = Node<MctsNodeData<N>, N>;
/// Type alias for a strong reference (`Rc<RefCell<...>>`) to an `MctsNode`.
type MctsNodeRef<const N: usize> = NodeRef<MctsNodeData<N>, N>;

/// Represents the current state of an MCTS (Monte Carlo Tree Search) instance,
/// controlling the flow of operations and preventing invalid sequential calls.
#[derive(Clone, Debug, PartialEq)]
pub enum MctsState{
    /// The MCTS instance is in a normal, ready-to-use state.
    /// All normal operations can be performed.
    Usable,
    /// The MCTS instance is awaiting the result of an external simulation.
    /// Only `apply_simulation` can be called in this state.
    AwaitingSimulation,
    /// The MCTS instance is temporarily locked during an internal operation
    /// (e.g., selection, expansion, backpropagation).
    /// No public methods should be called while in this state.
    Locked
}

/// Represents possible errors that can occur during MCTS (Monte Carlo Tree Search) operations.
#[derive(Debug)]
pub enum MctsError{
    /// Indicates that an MCTS operation was attempted when the instance was not in the
    /// required state (e.g., calling `apply_simulation` without `start_iteration`,
    /// or calling any method while in `Locked` state).
    InvalidState(MctsState),
    /// Occurs when the number of provided evaluations does not match the expected count
    /// (e.g., in `MctsBatch::apply_simulation`).
    /// Contains (expected_count, received_count).
    InvalidEvaluationCount(usize, usize),
    //// Indicates that an MCTS search cannot proceed because the root node
    /// already represents a finished game state. Further iterations or plays are
    /// not possible.
    SearchAlreadyOver,
    /// Occurs when an action index provided is outside the valid range [0, N-1] for the game.
    /// Contains the (attempted_action_index, max_action_index_N).
    ActionOutOfRange(usize, usize),
    /// Indicates that a chosen action is invalid according to the game's mask.
    /// This means the game state does not allow this action.
    /// Contains the invalid action index.
    InvalidAction(usize),
    /// Occurs when an action cannot be checked or played because the root
    /// node is none, it has not yet been explored and added to the MCTS tree.
    UnexploredAction
}

/// Type alias for a function pointer used to determine a node's selection score during MCTS.
///
/// This function takes the following parameters:
/// - `value`: The current mean value of the node.
/// - `policy`: The initial policy probability for the action leading to this node (from the parent's perspective).
/// - `n_visits`: The number of times the node has been visited.
/// - `parent_n_visits`: The number of times the parent node has been visited.
/// - `exploration_coef`: The exploration coefficient from `MctsConfig`.
///
/// It returns an `f64` score used to rank nodes for selection.
pub type SelectionFunction<const N: usize> = fn(value: f64, policy: f64, n_visits: f64, parent_n_visits: f64, exploration_coef: f64) -> f64;

/// Configuration parameters for a single Monte Carlo Tree Search (MCTS) instance.
///
/// This struct allows customization of MCTS behavior, including the exploration-exploitation
/// balance and the specific function used to calculate node selection scores.
///
/// # Type Parameters
/// - `N`: The number of possible actions in the game.
pub struct MctsConfig<const N: usize>{
    /// The exploration coefficient (often denoted as C_p or C_u) used in the selection phase.
    ///
    /// A higher value encourages more exploration of less-visited nodes, while a lower value
    /// prioritizes exploitation of known good paths.
    pub exploration_coef: f64,
    /// The function used to calculate the selection score for a child node during MCTS traversal.
    ///
    /// This function typically balances exploitation (based on value) and exploration (based on visits).
    /// You can use provided functions like `ucb1` or `default_selection_score`, or define your own.
    pub selection_function: SelectionFunction<N>
}

impl<const N: usize> MctsConfig<N>{
    /// The default MCTS configuration.
    ///
    /// - `exploration_coef`: `std::f64::consts::SQRT_2` (approximately 1.414), a common choice for UCB1.
    /// - `selection_function`: `default_selection_score`, the default formula.
    pub const DEFAULT: MctsConfig<N> = MctsConfig{
        exploration_coef: std::f64::consts::SQRT_2,
        selection_function: default_selection_score::<N>
    };
}

/// The Monte Carlo Tree Search algorithm implementation.
///
/// This struct manages the MCTS tree for a single game instance, allowing
/// for iterative search, game progression, and result retrieval.
///
/// # Type Parameters
/// - `T`: The game type that implements the `Game` trait.
/// - `N`: The number of possible actions in the game, a constant generic.
pub struct Mcts<T: Game<N>, const N: usize>{
    game: T,
    root: Option<MctsNodeRef<N>>,
    coef: f64,
    state: MctsState,
    /// Stores intermediate state between start_iteration and apply_simulation
    /// Contains: (game_state, node_to_simulate)
    latent: Option<(T, MctsNodeRef<N>)>,
    selection_function: SelectionFunction<N>
}

/// A selection function that combines value, visit count, and initial policy.
///
/// This function prioritizes unvisited nodes. For visited nodes, it balances exploitation
/// (node's value) with an exploration term that incorporates the initial policy
/// probability.
///
/// # Parameters
/// - `value`: The mean value of the node.
/// - `policy`: The initial policy probability for the action leading to this node.
/// - `n_visits`: Number of visits to the current node.
/// - `parent_n_visits`: Number of visits to the parent node.
/// - `exploration_coef`: The exploration coefficient.
///
/// # Returns
/// The calculated selection score for the node.
pub fn default_selection_score<const N: usize>(value: f64, policy: f64, n_visits: f64, parent_n_visits: f64, exploration_coef: f64) -> f64{
    value + exploration_coef * policy * parent_n_visits.sqrt() / (1.+n_visits)
}

/// The standard Upper Confidence Bound 1 (UCB1) selection function.
///
/// This function balances exploitation (current value) and exploration (unvisited nodes or less-visited nodes).
///
/// # Parameters
/// - `value`: The mean value (exploitation term) of the node.
/// - `policy`: This parameter is ignored in the standard UCB1 formula, but is present to match `SelectionFunction` signature.
/// - `n_visits`: Number of visits to the current node.
/// - `parent_n_visits`: Number of visits to the parent node.
/// - `exploration_coef`: The exploration coefficient.
///
/// # Returns
/// The UCB1 score for the node.
pub fn ucb1<const N: usize>(value: f64, policy: f64, n_visits: f64, parent_n_visits: f64, exploration_coef: f64) -> f64{
    value + exploration_coef * policy * (parent_n_visits.ln() / n_visits).sqrt()
}

impl<T: Game<N>, const N: usize> Mcts<T, N>{
    /// Standard score representing a victory in the game (e.g., for the current player).
    pub const VICTORY_SCORE: f64 = 1.0;
    /// Standard score representing a defeat in the game (e.g., for the current player).
    pub const DEFEAT_SCORE: f64 = -1.0;
    /// Standard score representing a draw or tie in the game.
    pub const EQUALITY_SCORE: f64 = 0.0;

    /// Creates a new MCTS instance with the default configuration.
    ///
    /// The default configuration uses `MctsConfig::DEFAULT`, which includes a
    /// standard `exploration_coef` and the `default_selection_score` selection function.
    ///
    /// # Returns
    /// A new MCTS instance ready to start searching from a new game.
    #[inline]
    pub fn new() -> Self{
        Self::from_config(&MctsConfig::DEFAULT)
    }

    /// Creates a new MCTS instance from a specified configuration.
    ///
    /// This allows users to customize the exploration coefficient and the selection
    /// function used during the MCTS process.
    ///
    /// # Parameters
    /// - `config`: The `MctsConfig` to use for this instance.
    ///
    /// # Returns
    /// A new MCTS instance initialized with the given configuration, ready to start
    /// searching from a new game.
    #[inline]
    pub fn from_config(config: &MctsConfig<N>) -> Self{
        Self::from_game_with_config(T::new(), config)
    }

    /// Creates a new MCTS instance starting from an existing game state with the default configuration.
    ///
    /// This is useful when you want to continue a search from a specific point in a game
    /// without custom MCTS parameters.
    ///
    /// # Parameters
    /// - `game`: The initial game instance.
    ///
    /// # Returns
    /// A new MCTS instance rooted at the given game state, using `MctsConfig::DEFAULT`.
    #[inline]
    pub fn from_game(game: T) -> Self{
        Mcts::from_game_with_config(game, &MctsConfig::DEFAULT)
    }

    /// Creates a new MCTS instance starting from an existing game state with a custom configuration.
    ///
    /// This allows resuming a search from a specific game point with fine-tuned MCTS parameters.
    ///
    /// # Parameters
    /// - `game`: The initial game instance.
    /// - `config`: The `MctsConfig` to use for this instance.
    ///
    /// # Returns
    /// A new MCTS instance rooted at the given game state, initialized with the provided configuration.
    #[inline]
    pub fn from_game_with_config(game: T, config: &MctsConfig<N>) -> Self{
        Mcts { 
            game: game, 
            root: None, 
            coef: config.exploration_coef, 
            state: MctsState::Usable, 
            latent: None, 
            selection_function: config.selection_function
        }
    }

    /// Gets an immutable reference to the underlying game instance.
    ///
    /// This allows inspection of the game state without modifying the MCTS tree.
    ///
    /// # Returns
    /// A reference to the internal `Game` instance.
    pub fn get_game(&self) -> &T{
        &self.game
    }

    /// Returns the current operational state of the MCTS instance.
    ///
    /// This indicates whether the MCTS is ready for a new iteration, awaiting
    /// simulation results, or temporarily locked.
    ///
    /// # Returns
    /// A clone of the current `MctsState`.
    #[inline]
    pub fn get_state(&self) -> MctsState{
        self.state.clone()
    }

    /// Calculates the UCB1 score for a child node during the selection phase.
    ///
    /// This private helper function computes the Upper Confidence Bound 1 (UCB1)
    /// value for a specific child of a given parent node. It's used to balance
    /// exploration and exploitation in MCTS tree traversal.
    ///
    /// # Parameters
    /// - `node`: The parent `MctsNode` from which the child originates.
    /// - `index`: The index of the child (representing an action) for which to calculate the score.
    ///
    /// # Returns
    /// The calculated UCB1 score (`f64`). Returns `-INFINITY` if the child node
    /// represents a finished game state, to avoid selecting it for further expansion.
    /// Returns INFINITY ponderate by policy for unexplorated node for keep order.
    #[inline]
    fn get_selection_score(&self, node: &MctsNode<N>, index: usize) -> f64{
        if let Some(child) = node.get_child(index){
            let node_child = &*child.borrow();

            if node_child.get().is_finish() { 
                -INFINITY 
            }
            else{
                (self.selection_function) (
                    node_child.get().get_value(), 
                    node.get().get_policy(index), 
                    node_child.get().get_n() as f64, 
                    node.get().get_n() as f64, 
                    self.coef
                )
            }
        }
        else{
            if node.get().get_mask(index) && !node.get().is_finish() { INFINITY * (1. + node.get().get_policy(index))} else { -INFINITY }
        }
    }

    /// Performs the selection phase of MCTS
    ///
    /// # Returns
    /// Tuple containing:
    /// - The selected node
    /// - The action taken to reach it
    /// - The game state at that node
    #[inline]
    fn selection(&self) -> (Option<MctsNodeRef<N>>, usize, T){
        let mut game: T = self.game.clone();

        let mut node = match &self.root {
            Some(root) => Rc::clone(root),
            None => return (None, 0, game)
        };

        loop {
            let scores : [f64; N] = std::array::from_fn(|index| self.get_selection_score(&node.borrow(), index));

            let index = scores.iter().enumerate().max_by(|a, b| (a.1).total_cmp(b.1)).unwrap().0;
            game.play(index);

            if node.borrow().get_child(index).is_none() {
                return (Some(node), index, game);
            }

            let next = node.borrow().get_child(index).unwrap();
            node = next;
        }
    }

    /// Performs the expansion phase of MCTS
    ///
    /// # Parameters
    /// - `node`: The node to expand from
    /// - `index`: The action to expand
    ///
    /// # Returns
    /// The newly created child node
    #[inline]
    fn expansion(&mut self, node: &Option<MctsNodeRef<N>>, index: usize) -> MctsNodeRef<N>{
        if let Some(node) = node{
            Node::add_child(&node, index, MctsNodeData::new())
        }
        else{
            let node_ref = Rc::new(RefCell::new(
                MctsNode::new(None, MctsNodeData::new())
            ));

            self.root = Some(Rc::clone(&node_ref));
            node_ref
        }
    }

    /// Performs the simulation phase of MCTS
    ///
    /// # Parameters
    /// - `node`: The node to simulate from
    /// - `game`: The game state at that node
    /// - `evaluator`: The policy/value evaluator
    #[inline]
    fn simulation(&mut self, node: &mut MctsNode<N>, game: &T, evaluator: &dyn GameEvaluator<T, N>){
        let data = node.get_mut();

        if let Some(score) = game.get_result(){
            data.add_score(-score);
            data.finish = true;
        }
        else{
            let (score, policy) = evaluator.evaluate(game.get_state());

            data.add_score(-score);
            data.policy=policy;
            data.mask=game.get_actions();
        }
    }

    /// Performs simulation using precomputed evaluation data
    ///
    /// # Panics
    /// If called when not in `AwaitingSimulation` state
    ///
    /// # Parameters
    /// - `node`: The node to simulate from
    /// - `game`: The game state at that node
    /// - `evaluation`: The policy/value
    #[inline]
    fn simulation_from_data(&mut self, node: &mut MctsNode<N>, game: &T, evaluation: (f64, [f64; N])){
        let data = node.get_mut();

        if let Some(score) = game.get_result(){
            data.add_score(-score);
            data.finish = true;
        }
        else{
            let (score, policy) = evaluation;

            data.add_score(-score);
            data.policy=policy;
            data.mask=game.get_actions();
        }
    }

    /// Performs the backpropagation phase of MCTS
    ///
    /// # Parameters
    /// - `node_ref`: The node to start backpropagation from
    #[inline]
    fn backpropagation(&mut self, node_ref: &MctsNodeRef<N>){
        let mut current_ref_opt: Option<Rc<RefCell<Node<MctsNodeData<N>, N>>>>;

        let mut score: f64;
        let mut finish: bool;

        {
            let node = &*node_ref.borrow();
            score = node.get().get_value();
            finish = node.get().is_finish();
            current_ref_opt = node.get_parent();
        }

        while let Some(current_ref) = current_ref_opt {
            score = -score;
            
            let current = &mut *current_ref.borrow_mut();

            if !finish{
                current.get_mut().add_score(score);
            }
            else {
                //the score is already inverse Defeat => Victory and Victory => Defeat
                if score == -Self::VICTORY_SCORE {
                    current.get_mut().score = current.get().n as f64 * score;
                    current.get_mut().finish = true;
                }
                else{
                    let mut max_score = f64::MIN;
                    let current_is_finish = (0..N).all(|index|{
                        !current.get().get_mask(index) || {
                            if let Some(child_ref) = current.get_child(index){
                                let child = &*child_ref.borrow();

                                let value = child.get().get_value();
                                if value > max_score {
                                    max_score = value;
                                }

                                child.get().is_finish()
                            }
                            else{ false }
                        }
                    });

                    if current_is_finish {
                        current.get_mut().score = current.get().n as f64 * -max_score;
                        current.get_mut().finish = true;
                    }
                    else{
                        current.get_mut().add_score(score);
                        finish = false;
                    }
                }
            }
            
            current_ref_opt = current.get_parent();
        }
    }

    /// Performs one full iteration of MCTS (selection, expansion, simulation, backpropagation).
    ///
    /// This method requires the MCTS instance to be in a `Usable` state.
    ///
    /// # Parameters
    /// - `evaluator`: The policy/value evaluator to use.
    ///
    /// # Returns
    /// `Ok(())` if the iteration completes successfully.
    /// `Err(MctsError::InvalidState(_))` if the MCTS instance is not in the `Usable` state.
    #[inline]
    pub fn iterate(&mut self, evaluator: &dyn GameEvaluator<T, N>) -> Result<(), MctsError> {
        if self.state != MctsState::Usable{
            return Err(MctsError::InvalidState(self.state.clone()))
        }

        self.state = MctsState::Locked;

        if let Some(root) = self.root.as_ref(){
            if root.borrow().get().is_finish(){
                self.state = MctsState::Usable;
                return Ok(());
            }
        }

        let (node, index, game) = self.selection();
        let child_ref = self.expansion(&node, index);
        self.simulation(&mut *child_ref.borrow_mut(), &game, evaluator);
        self.backpropagation(&child_ref);

        self.state = MctsState::Usable;
        Ok(())
    }

    /// Performs the first partial iteration of MCTS (selection and expansion),
    /// returning the game state for external simulation.
    ///
    /// This method transitions the MCTS instance from `Usable` to `AwaitingSimulation` state.
    ///
    /// # Returns
    /// `Ok(game_state)` containing the game state requiring evaluation.
    /// `Err(MctsError::InvalidState(_))` if the MCTS instance is not in the `Usable` state.
    /// `Err(MctsError::SearchAlreadyOver)` if the root node already represents a finished game.
    #[inline]
    pub fn start_iteration(&mut self) -> Result<T::State, MctsError>{
        if self.state != MctsState::Usable{
            return Err(MctsError::InvalidState(self.state.clone()));
        }

        self.state = MctsState::Locked;

        if let Some(root) = self.root.as_ref(){
            if root.borrow().get().is_finish(){
                return Err(MctsError::SearchAlreadyOver)
            }
        }

        let (node, index, game) = self.selection();
        let child_ref = self.expansion(&node, index);

        let game_state = game.get_state();

        self.latent = Some((game, child_ref));
        self.state = MctsState::AwaitingSimulation;

        Ok(game_state)
    }

    /// Completes a partial MCTS iteration by applying an external simulation's evaluation
    /// and performing backpropagation.
    ///
    /// This method transitions the MCTS instance from `AwaitingSimulation` back to `Usable` state.
    ///
    /// # Parameters
    /// - `evaluation`: A tuple containing the estimated value (f64) and action probabilities ([f64; N])
    ///                 from the external simulation.
    ///
    /// # Returns
    /// `Ok(())` if the simulation is successfully applied and backpropagation completes.
    /// `Err(MctsError::InvalidState(_))` if the MCTS instance is not in the `AwaitingSimulation` state.
    #[inline]
    pub fn apply_simulation(&mut self, evaluation : (f64, [f64; N])) -> Result<(), MctsError>{
        if self.state != MctsState::AwaitingSimulation || self.latent.is_none() {
            return Err(MctsError::InvalidState(self.state.clone()));
        }
        self.state = MctsState::Locked;

        let (game, child_ref) = self.latent.take().unwrap();

        self.simulation_from_data(&mut *child_ref.borrow_mut(), &game, evaluation);
        self.backpropagation(&child_ref);

        self.state = MctsState::Usable;
        Ok(())
    }

    /// Determines if the MCTS search has concluded, either because the game is finished
    /// or due to other stopping criteria (though currently only checks for game finish).
    ///
    /// # Returns
    /// `true` if the MCTS search is considered finished (e.g., game over at root), `false` otherwise.
    #[inline]
    pub fn is_finish(&self) -> bool{
        if let Some(root) = &self.root{
            root.borrow().get().is_finish()
        }
        else{ false }
    }

    /// Gets the current value estimate for the root node
    #[inline]
    pub fn get_score(&self) -> f64{
        if let Some(root) = &self.root {
            -root.borrow().get().get_value()
        }
        else{ Self::EQUALITY_SCORE }
    }

    /// Calculates action probabilities from the root node's statistics
    ///
    /// # Parameters
    /// - `root`: The root node to calculate from
    ///
    /// # Returns
    /// Array of action probabilities
    #[inline]
    fn statistics_from_root(root: &MctsNode<N>) -> [f64; N]{
        /*
            We project the values of the nodes from the interval ]-1; 1[ to ]-inf; +inf[
            using the function ln((1+x)/(1-x)). To calculate probabilities, we use softmax,
            so the logarithm simplifies and the function becomes (1+x)/(1-x).
            The function ln((1+x)/(1-x)) is the reciprocal of tanh(x/2).
        */

        // 0 -> score = -inf
        let scores: [f64; N] = std::array::from_fn(|index|{
            if let Some(child_ref) = root.get_child(index){
                let score = child_ref.borrow().get().get_value();

                if score != 1.{ (score + 1.) / (1. - score) + f64::MIN_POSITIVE } else{ f64::MAX / N as f64 }
            }
            else if root.get().get_mask(index){ f64::MIN_POSITIVE }
            else { 0.0 }
        });

        let total: f64 = scores.iter().sum();
        let total: f64 = if total == 0.0 { f64::MIN_POSITIVE } else{ total };

        let scores = scores.map(|x| x/total);

        scores
    }

    /// Gets the current action probabilities from the root node
    #[inline]
    pub fn get_statistics(&self) -> [f64; N]{
        if let Some(root_ref) = &self.root {
            Self::statistics_from_root(&*root_ref.borrow())
        }
        else{
            [1./N as f64; N]
        }
    }

    /// Returns the final result of the MCTS search (best score and policy).
    ///
    /// This method is typically called when the search is considered complete
    /// or when a decision needs to be made based on the current tree.
    ///
    /// # Returns
    /// A tuple containing:
    /// - The average value of the root node (`f64`).
    /// - An array of action probabilities ([f64; N]), which is usually the policy
    ///   from the root node adjusted by visit counts for robust decision making.
    #[inline]
    pub fn get_result(&self) -> (f64, [f64; N]){
        if let Some(root_ref) = &self.root {
            let root = &*root_ref.borrow();
            (-root.get().get_value(), Self::statistics_from_root(root))
        }
        else{
            (0.0, [1./N as f64; N])
        }
    }

    /// Calculates the total number of visits across all nodes in the MCTS tree.
    ///
    /// This can be used as a metric for the extent of the search performed.
    ///
    /// # Returns
    /// The sum of visit counts (`n`) of all nodes in the tree.
    #[inline]
    pub fn count_visit(&self) -> usize{
        if let Some(root_ref) = &self.root{
            let root = &*root_ref.borrow();
            root.get().get_n()
        }
        else { 0 }
    }

    /// Moves the MCTS root to the specified child, effectively playing an action.
    ///
    /// This method prunes the tree, discarding all branches not descending from the chosen child.
    /// The MCTS instance must be in a `Usable` state.
    ///
    /// # Parameters
    /// - `action`: The index of the child (action) to play.
    ///
    /// # Returns
    /// `Ok(())` if the root is successfully moved to the child corresponding to the action.
    /// `Err(MctsError::InvalidState(_))` if the MCTS instance is not in the `Usable` state.
    /// `Err(MctsError::ActionOutOfRange(action, N))` if the `action` index is out of bounds (>= N).
    /// `Err(MctsError::InvalidAction(_))` if the action is invalid according to the game mask.
    /// `Err(MctsError::UnexploredAction)` if the action cannot be check because root is null.
    #[inline]
    pub fn play(&mut self, action: usize) -> Result<(), MctsError>{
        if self.state != MctsState::Usable {
            return Err(MctsError::InvalidState(self.state.clone()));
        }

        if action >= N {
            return Err(MctsError::ActionOutOfRange(action, N));
        }

        self.state = MctsState::Locked;

        let new_root;

        if let Some(root_ref) = &self.root{
            let root = &*root_ref.borrow();

            if let Some(child_ref) = root.get_child(action){
                {
                    let child = &mut *child_ref.borrow_mut();
                    child.detach();
                }

                new_root = Some(child_ref);
            }
            else if root.get().get_mask(action){
                new_root = None;
            }
            else{
                return Err(MctsError::InvalidAction(action));
            }
        }
        else{
            return Err(MctsError::UnexploredAction);
        }

        self.game.play(action);
        self.root = new_root;

        self.state = MctsState::Usable;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::{test_utils::{compare_array, GameEvaluatorTest, GameEvaluatorTest2, GameTest}, Game, GameEvaluator, Mcts, MctsError};

    #[test]
    fn test_selection_empty(){
        let mcts = Mcts::<GameTest, 4>::new();
        assert!(mcts.selection().0.is_none())
    }

    #[test]
    fn test_expansion_empty(){
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        mcts.expansion(&None, 0);

        assert!(mcts.root.is_some());
    }

    #[test]
    fn test_selection_root(){
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        mcts.expansion(&None, 0);

        let (node, _index, _game)  = mcts.selection();
        let node = &*node.as_ref().unwrap().borrow();

        assert!(node.is_root());

        assert!(node.get_child(0).is_none());
        assert!(node.get_child(1).is_none());
        assert!(node.get_child(2).is_none());
        assert!(node.get_child(3).is_none());
    }

    #[test] 
    fn test_simulation_root(){
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest = GameEvaluatorTest::new();

        let (node, index, game) = mcts.selection();
        let child = mcts.expansion(&node, index);
        let child = &mut *child.borrow_mut();

        mcts.simulation(child, &game, &evaluator);

        assert!(child.is_root());
        assert!(child.get_child(0).is_none());
        assert!(child.get_child(1).is_none());
        assert!(child.get_child(2).is_none());
        assert!(child.get_child(3).is_none());

        assert!(!child.get().is_finish());
        assert_eq!(child.get().get_policy(0), 0.2);
        assert_eq!(child.get().get_policy(1), 0.7);
        assert_eq!(child.get().get_policy(2), 0.06);
        assert_eq!(child.get().get_policy(3), 0.04);
    }

    #[test]
    fn test_iteration_empty() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest = GameEvaluatorTest::new();

        mcts.iterate(&evaluator)?;

        {
            let node_ref = mcts.root.as_ref().unwrap();
            let node = &*node_ref.borrow();

            assert!(node.is_root());
            assert!(!node.get().is_finish());
            assert_eq!(node.get().mask, [true, true, true, true]);
            assert_eq!(node.get().policy, [0.2, 0.7, 0.06, 0.04]);
            assert_eq!(node.get().score, 0.0);
            assert_eq!(node.get().n, 1);
        }

        assert!(!mcts.is_finish());
        Ok(())
    }

    #[test]
    fn test_iteration_root_with_no_child() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest = GameEvaluatorTest::new();

        mcts.iterate(&evaluator)?;
        mcts.iterate(&evaluator)?;

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert_eq!(root.get().score, 0.2);
            assert_eq!(root.get().n, 2);
            assert!(root.get_child(0).is_none());
            assert!(root.get_child(1).is_some());
            assert!(root.get_child(2).is_none());
            assert!(root.get_child(3).is_none());
        }

        mcts.iterate(&evaluator)?;
        mcts.iterate(&evaluator)?;
        mcts.iterate(&evaluator)?;

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert_eq!(root.get().score, 0.0);
            assert_eq!(root.get().n, 5);
            assert!(root.get_child(0).is_some());
            assert!(root.get_child(1).is_some());
            assert!(root.get_child(2).is_some());
            assert!(root.get_child(3).is_some());
        }

        assert!(!mcts.is_finish());
        Ok(())
    }

    #[test]
    fn test_iteration_root_with_child() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest = GameEvaluatorTest::new();

        for _ in 0..7{
            mcts.iterate(&evaluator)?;
        }

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert_eq!(root.get().score, 0.0);
            assert_eq!(root.get().n, 7);
        }

        assert!(!mcts.is_finish());
        Ok(())
    }

    #[test]
    fn test_iteration_victory_1() -> Result<(), MctsError>{
        let mut game: GameTest = GameTest::new();
        game.play(3);
        game.play(1);
        game.play(2);

        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::from_game(game);
        let evaluator: GameEvaluatorTest = GameEvaluatorTest::new();

        mcts.iterate(&evaluator)?;
        mcts.iterate(&evaluator)?;

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert!(root.get().is_finish());
            assert_eq!(root.get().get_value(), 1.0);
        }

        assert!(mcts.is_finish());
        assert_eq!(mcts.get_result(), (-1.0, [1.0, 0., 0., 0.]));
        Ok(())
    }

    #[test]
    fn test_iteration_victory_2() -> Result<(), MctsError>{
        let mut game: GameTest = GameTest::new();
        game.play(0);
        game.play(3);
        game.play(1);

        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::from_game(game);
        let evaluator: GameEvaluatorTest = GameEvaluatorTest::new();

        mcts.iterate(&evaluator)?;
        mcts.iterate(&evaluator)?;

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert!(root.get().is_finish());
            assert_eq!(root.get().get_value(), -1.0);
        }

        assert!(mcts.is_finish());
        assert_eq!(mcts.get_result(), (1.0, [0., 0., 1., 0.]));
        Ok(())
    }

    #[test]
    fn test_iteration_end_1() -> Result<(), MctsError>{
        let mut game: GameTest = GameTest::new();
        game.play(3);
        game.play(1);

        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::from_game(game);
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        for _ in 0..4{
            mcts.iterate(&evaluator)?;
        }

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert!(root.get().is_finish());
            assert_eq!(root.get().get_value(), -1.0);
        }

        assert!(mcts.is_finish());

        let result = mcts.get_result();
        assert_eq!(result.0, 1.0);
        assert!(compare_array(&result.1, &[0., 0., 1., 0.]));
        Ok(())
    }

    #[test]
    fn test_iteration_total() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        for _ in 0..18{
            mcts.iterate(&evaluator)?;
        }

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert!(root.get().is_finish());
            assert_eq!(root.get().get_value(), -1.0);
        }

        assert!(mcts.is_finish());

        let result = mcts.get_result();
        assert_eq!(result.0, 1.0);
        assert!(compare_array(&result.1, &[0., 0., 0., 1.]));
        Ok(())
    }

    #[test]
    fn test_iteration_total_2() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        for _ in 0..18{
            let game = mcts.start_iteration()?;
            mcts.apply_simulation(evaluator.evaluate(game))?;
        }

        {
            let root_ref = mcts.root.as_ref().unwrap();
            let root = &*root_ref.borrow();

            assert!(root.get().is_finish());
            assert_eq!(root.get().get_value(), -1.0);
        }

        assert!(mcts.is_finish());

        let result = mcts.get_result();
        assert_eq!(result.0, 1.0);
        assert!(compare_array(&result.1, &[0., 0., 0., 1.]));
        Ok(())
    }

    #[test]
    fn test_play_and_count_visit() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        assert_eq!(mcts.count_visit(), 0);

        for _ in 0..6{
            mcts.iterate(&evaluator)?;
        }

        assert_eq!(mcts.count_visit(), 6);
        mcts.play(3)?;
        assert_eq!(mcts.count_visit(), 2);

        assert!(mcts.root.unwrap().borrow().is_root());
        Ok(())
    }

    #[test]
    fn test_play_empty() -> Result<(), MctsError>{
        let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        mcts.iterate(&evaluator)?;
        mcts.play(3)?;
        Ok(())
    }
}
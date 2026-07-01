use crate::tree::*;

/// Represents an index corresponding to a specific legal move or action.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct Action(usize);

impl Action{
    /// Constructs a new `Action`.
    ///
    /// # Arguments
    /// * `action` - The integer index representing the move.
    ///
    /// # Returns
    /// A new `Action` instance.
    pub fn new(action: usize) -> Self{
        Action(action)
    }

    /// Returns the underlying integer index of the action.
    ///
    /// # Returns
    /// The `usize` value stored inside the `Action`.
    pub fn action(&self) -> usize{
        self.0
    }
}

/// Represents the internal tactical state of a node in the MCTS tree.
#[derive(Copy, Clone, Debug, PartialEq)]
enum MctsNodeState{
    /// The node is active and can be expanded or traversed further.
    Active,
    /// The node represents a terminal state with a fixed final game score.
    Terminal(f32)
}

/// Internal container representing a single state within the Monte Carlo Tree Search.
///
/// It utilizes a Structure of Arrays (SoA) layout for its children's statistics
/// (`children_scores`, `children_visits`, `children_policies`) to guarantee
/// cache-friendly memory lookups during the intensive selection phase.
#[derive(Debug, Copy, Clone, PartialEq)]
struct MctsNode<const N: usize>{
    children_scores: [f32; N],
    children_visits: [i32; N],
    children_policies: [f32; N],

    visits: i32,
    state: MctsNodeState,
    action: Action,
}


/// Evaluates a child node during the tree selection phase.
///
/// This function determines whether the current node should be prioritized.
/// It typically implements an exploration/exploitation
/// balance formula such as UCB1.
///
/// # Arguments
///
/// * `score` - The score value of the child node (w).
/// * `node_visits` - The number of visits to the currently evaluated child node (n).
/// * `parent_visits` - The total number of visits to the parent node (N).
/// * `policy` - The exploration policy value.
/// * `c` - The exploration constant used to adjust the ratio (often sqrt(2)).
///
/// # Returns
///
/// Returns a floating-point score representing the node's priority
pub trait SelectionFunction: Fn(f32, i32, i32, f32, f32) -> f32 {}
impl<T: Fn(f32, i32, i32, f32, f32) -> f32> SelectionFunction for T {}

/// Transforms a score during the backpropagation phase.
///
/// This is typically used in alternating-turn games (like Chess or Tic-Tac-Toe)
/// to invert the score (e.g., `|score| -score`) from one depth to another.
pub trait ScoreTransformer: FnMut(f32) -> f32 {}
impl<T: FnMut(f32) -> f32> ScoreTransformer for T {}

/// A buffer used to store the sequence of actions traversed during the selection phase.
pub trait ResettableBuffer {
    /// Pushes an action to the end of the buffer.
    fn push(&mut self, value: Action);
    /// Clears the buffer, removing all elements.
    fn clear(&mut self);
}

/// Implements `ResettableBuffer` for standard collections.
#[macro_export]
macro_rules! impl_resettable_buffer {
    ($type:ty) => {
        impl ResettableBuffer for $type {
            fn push(&mut self, value: Action) {
                self.push(value);
            }
            fn clear(&mut self) {
                self.clear();
            }
        }
    };
}

impl_resettable_buffer!(Vec<Action>);

impl<const N: usize> MctsNode<N> {
    /// Constructs a new, unvisited `MctsNode` with prior policies, an incoming action, and a state.
    fn new(policies: [f32; N], action: Action, state: MctsNodeState) -> Self {
        MctsNode{
            children_scores: [0.; N],
            children_visits:  [0; N],
            children_policies: policies,
            visits: 0,
            action,
            state
        }
    }

    /// Selects the best child action index based on the provided selection heuristic.
    ///
    /// It systematically filters out invalid actions (where policy <= 0.0) and uses
    /// `total_cmp` to safely sort floating-point numbers.
    ///
    /// # Panics
    ///
    /// Panics if no legal actions remain (i.e., all policies are less than or equal to 0.0),
    /// indicating a logical mismatch with the game wrapper's state.
    fn best_child(&self, score_f: &impl SelectionFunction, c: f32) -> usize {
        self.children_scores.iter()
            .zip(self.children_visits.iter())
            .zip(self.children_policies.iter())
            .enumerate()
            .filter_map(
            |(i, ((c_scores, c_visits), policy))| {
                if *policy <= 0. { None } else { Some((i, score_f(*c_scores, *c_visits, self.visits, *policy, c))) }
            }
        ).max_by(
            |(_a, a), (_b, b)| { a.total_cmp(b) }
        ).expect("Error during the selection of the best action.").0
    }
}

/// A strictly typed identifier for an instantiated node within the MCTS tree.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct MctsNodeId(NodeId);

/// The core Engine handling the Monte Carlo Tree Search algorithm.
///
/// It manages the internal tree structure and provides methods to traverse,
/// expand, and backpropagate statistics.
pub struct Engine<const N: usize> {
    tree: Tree<MctsNode<N>, N>
}

/// Represents the outcome of a tree selection phase.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum SelectionResult {
    /// The tree is entirely empty.
    Empty,
    /// A leaf node was reached. Provides the parent ID and the selected action.
    Active(MctsNodeId, Action),
    /// A terminal node was reached during traversal. Provides the node ID and final score.
    Terminal(MctsNodeId, f32)
}

/// Represents the evaluation of a game state, usually provided by a heuristic function
/// or a neural network (e.g., in AlphaZero/MuZero architectures).
///
/// This enum strictly binds the game's termination status with its corresponding data,
/// preventing the representation of invalid states (like a terminal state having future policies).
///
/// # Note on Score Perspective
/// The `score` value provided here (both in `Active` and `Terminal` states) must
/// always be expressed from the perspective of the player who **just made the move**
/// that led to this state. This ensures correct backpropagation in alternating-turn games
/// when using the engine's `score_updater` logic.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum StateEvaluation<const N: usize> {
    /// The game is ongoing. Contains the current board evaluation score and the prior
    /// probabilities (policies) for the next `N` possible actions.
    Active(f32, [f32; N]),

    /// The game has ended. Contains only the final fixed score (e.g., 1.0 for win, -1.0 for loss).
    Terminal(f32)
}

impl<const N: usize> StateEvaluation<N>{
    /// Extracts the score from the evaluation, regardless of whether the state is active or terminal.
    ///
    /// # Returns
    ///
    /// A floating-point value (`f32`) representing the underlying score of this state.
    pub fn score(&self) -> f32{
        match self {
            StateEvaluation::Active(score, _) => *score,
            StateEvaluation::Terminal(score) => *score
        }
    }
}

/// Errors that can occur during the execution of the MCTS engine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MctsEngineError{
    UnknownError,
    InvalidMctsNode,
    ChildAlreadyExists,
    InvalidAction,
    SelectionIsTerminal
}

impl<const N: usize> Engine<N> {
    /// Creates a new, empty MCTS engine.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::Engine;
    /// let mut engine = Engine::<7>::new();
    /// ```
    pub fn new() -> Self {
        Engine::<N> {
            tree: Tree::<MctsNode<N>, N>::new()
        }
    }

    /// Creates a new MCTS engine with a pre-allocated capacity for the underlying tree.
    ///
    /// This is recommended to avoid reallocations when running millions of simulations.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::Engine;
    /// let mut engine = Engine::<7>::with_capacity(1_000_000);
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Engine::<N> {
            tree: Tree::<MctsNode<N>, N>::with_capacity(capacity)
        }
    }

    /// Traverses the tree from the root to a leaf node according to the selection function.
    ///
    /// The traversal path (sequence of actions) is recorded in `path_out`.
    ///
    /// # Arguments
    ///
    /// * `path_out` - A mutable reference to a buffer that will be cleared and filled with the chosen actions.
    /// * `score_f` - The selection heuristic (e.g., PUCT formula).
    /// * `c` - The exploration hyperparameter.
    ///
    /// # Returns
    ///
    /// A `SelectionResult` indicating whether a new leaf was found, a terminal state was hit, or the tree is empty.
    ///
    /// # Panics
    ///
    /// Panics if the internal tree structure is corrupted (i.e., a parent node ID points to a non-existent element).
    /// Also panics if an active node has no legal moves (all policies are <= 0.0).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::Engine;
    /// # use crate::simple_mcts::puct;
    /// let mut path = Vec::new();
    /// let mut engine = Engine::<7>::new();
    /// let result = engine.select(&mut path, &puct, 1.414);
    /// ```
    pub fn select(&self, path_out: &mut impl ResettableBuffer, score_f: &impl SelectionFunction, c: f32) -> SelectionResult{
        path_out.clear();

        if let Some(mut current_id) = self.tree.root(){
            loop {
                let current = self.tree.get(current_id).unwrap();

                match current.data().state {
                    MctsNodeState::Terminal(score) => {
                        return SelectionResult::Terminal(MctsNodeId(current_id), score);
                    }
                    MctsNodeState::Active => {
                        let action = current.data().best_child(score_f, c);

                        path_out.push(Action(action));
                        if let Some(child_id) = current.child(action){
                            current_id = child_id;
                        }
                        else{
                            return SelectionResult::Active(MctsNodeId(current_id), Action(action));
                        }
                    }
                }
            }
        }
        else{ SelectionResult::Empty }
    }

    /// Expands the tree by attaching a new child node to the result of a previous selection.
    ///
    /// # Arguments
    ///
    /// * `evaluation` - The evaluated data of the new state (score and policies if active, or just the final score if terminal).
    /// * `selection` - The `SelectionResult` obtained from a previous call to `select`.
    ///
    /// # Returns
    ///
    /// Returns the newly minted `MctsNodeId` on success, or an `MctsEngineError`.
    ///
    /// # Panics
    ///
    /// Panics if the `SelectionResult` contains an out-of-bounds action index (`>= N`).
    /// Because valid selections are exclusively generated by the engine's `select` method,
    /// this panic indicates that the provided `SelectionResult` was either manually forged,
    /// corrupted, or used entirely out of context.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::{Engine, StateEvaluation};
    /// # use crate::simple_mcts::puct;
    /// let mut path = Vec::new();
    /// let mut engine = Engine::<2>::new();
    /// let result = engine.select(&mut path, &puct, 1.414);
    ///
    /// // Expand an ongoing game state
    /// let evaluation = StateEvaluation::Active(-0.2, [0.3, 0.7]);
    /// engine.expand(evaluation, result).unwrap();
    /// ```
    pub fn expand(&mut self, evaluation: StateEvaluation<N>, selection: SelectionResult) -> Result<MctsNodeId, MctsEngineError>{
        let (state, policies) = match evaluation {
            StateEvaluation::Active(_, policies) => (MctsNodeState::Active, policies),
            StateEvaluation::Terminal(score) => (MctsNodeState::Terminal(score), [0.; N])
        };

        match selection {
            SelectionResult::Empty => {
                match self.tree.set_root(MctsNode::new(policies, Action(0), state)) {
                    Err(TreeError::RootAlreadyExists) => Err(MctsEngineError::ChildAlreadyExists),
                    Err(_) => Err(MctsEngineError::UnknownError),
                    Ok(node) => Ok(MctsNodeId(node)),
                }
            },
            SelectionResult::Active(child_id, action) => {
                match self.tree.add(child_id.0, action.0, MctsNode::new(policies, action, state)) {
                    Err(TreeError::ParentDoesntExist) => Err(MctsEngineError::InvalidMctsNode),
                    Err(TreeError::ChildAlreadyExists) => Err(MctsEngineError::ChildAlreadyExists),
                    Err(_) => Err(MctsEngineError::UnknownError),
                    Ok(node) => Ok(MctsNodeId(node))
                }
            },
            SelectionResult::Terminal(_, _) => Err(MctsEngineError::SelectionIsTerminal)
        }
    }

    /// Backpropagates a simulation score up to the root of the tree.
    ///
    /// Every node encountered on the way up will have its visits incremented and
    /// its children statistics updated. The score is transformed at every step using `score_updater`.
    ///
    /// # Arguments
    ///
    /// * `node` - The `MctsNodeId` from which to start the backpropagation (usually the newly expanded leaf).
    /// * `score` - The initial score to propagate.
    /// * `score_updater` - A function or closure applied to the score at each step (e.g., negating it).
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` on success, or an error if an invalid node ID is encountered.
    ///
    /// # Examples
    ///
    /// ```
    /// # use simple_mcts::StateEvaluation;
    /// use crate::simple_mcts::Engine;
    /// # use crate::simple_mcts::puct;
    /// let mut path = Vec::new();
    /// let mut engine = Engine::<2>::new();
    /// let result = engine.select(&mut path, &puct, 1.414);
    /// let evaluation = StateEvaluation::Active(-0.2, [0.3, 0.7]);
    ///
    /// // Invert the score at each step for an alternating-turn game
    /// let node = engine.expand(evaluation, result).unwrap();
    /// engine.backpropagate(node, 1.0, |s| -s).unwrap();
    /// ```
    pub fn backpropagate(&mut self, node: MctsNodeId, score: f32, mut score_updater: impl ScoreTransformer) -> Result<(), MctsEngineError> {
        let mut current = Some(node.0);
        let mut score = score;

        while let Some(node) = current {
            let node = self.tree.get_mut(node).map_err(|_| MctsEngineError::InvalidMctsNode)?;
            node.data_mut().visits += 1;

            let action = node.data().action.0;
            current = node.parent();

            if let Some(parent) = current {
                let parent = self.tree.get_mut(parent).map_err(|_| MctsEngineError::InvalidMctsNode)?;
                parent.data_mut().children_visits[action] += 1;
                parent.data_mut().children_scores[action] += score;
            }

            score = score_updater(score);
        }

        Ok(())
    }

    /// A convenience method that sequentially performs `expand` and `backpropagate`.
    ///
    /// If the selection provided was already terminal, expansion is skipped and
    /// backpropagation starts immediately from the terminal node.
    ///
    /// # Arguments
    ///
    /// * `evaluation` - The evaluated data of the leaf node (score and policies).
    /// * `selection` - The result returned by a prior `select` call.
    /// * `score_updater` - The transformation applied to the score during backpropagation (e.g., inverting it).
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::{Engine, StateEvaluation};
    /// # use crate::simple_mcts::puct;
    /// let mut path = Vec::new();
    /// let mut engine = Engine::<2>::new();
    /// let result = engine.select(&mut path, &puct, 1.414);
    ///
    /// // Expand and backpropagate in one step, inverting the score at each depth
    /// let evaluation = StateEvaluation::Active(-0.2, [0.3, 0.7]);
    /// engine.update(evaluation, result, |s| -s).unwrap();
    /// ```
    pub fn update(&mut self, evaluation: StateEvaluation<N>, selection: SelectionResult, score_updater: impl ScoreTransformer) -> Result<(), MctsEngineError> {
        let score = evaluation.score();

        let node = match selection {
            SelectionResult::Empty => self.expand(evaluation, selection)?,
            SelectionResult::Active(_child_id, _action) => self.expand(evaluation, selection)?,
            SelectionResult::Terminal(child_id, _score) => child_id
        };

        self.backpropagate(node, score, score_updater)?;

        Ok(())
    }

    /// Retrieves the visit counts of all possible actions from the root node.
    ///
    /// This array is typically used at the end of the MCTS cycle to decide the actual
    /// move to play in the game.
    ///
    /// # Returns
    ///
    /// An array of integers representing the number of visits for each child branch.
    /// Returns an array of zeros if the tree is currently empty.
    ///
    /// # Panics
    ///
    /// Panics if the internal tree has a root ID but the root node cannot be retrieved.
    ///
    /// # Examples
    ///
    /// ```
    ///  # use crate::simple_mcts::Engine;
    /// let mut engine = Engine::<3>::new();
    /// let scores = engine.scores();
    /// assert_eq!(scores, [0, 0, 0]);
    /// ```
    pub fn scores(&self) -> [i32; N]{
        if let Some(root) = self.tree.root() {
            self.tree.get(root).unwrap().data().children_visits
        }
        else {
            [0; N]
        }
    }

    /// Promotes the child node corresponding to the given action to the new root of the tree.
    ///
    /// This method updates the MCTS tree to reflect a move played on the actual game board.
    /// It performs an amortized memory cleanup (compacting) if the number of unreachable
    /// nodes exceeds a heuristic threshold (twice the number of visits of the new root).
    ///
    /// If the action leads to a path that has not been explored by the MCTS, the current
    /// tree is cleared, as it no longer contains valid information for the new state.
    ///
    /// # Arguments
    ///
    /// * `action` - The action that was performed on the game board.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The action index is out of bounds (i.e., `action.0 >= N`).
    /// - The internal tree structure is corrupted (e.g., attempting to move the root to a non-existent node).
    ///
    /// # Note
    ///
    /// This is an amortized operation. `self.tree.compact()` is only called when
    /// memory overhead becomes significant, ensuring high performance during game play.
    pub fn commit_action(&mut self, action: Action) {
        if let Some(root) = self.tree.root() {
            let child = self.tree.child(root, action.0).unwrap();

            if let Some(new_root_id) = child {
                self.tree.move_root_to(new_root_id).unwrap();

                let new_root_data = self.tree.data(new_root_id).unwrap();
                if self.tree.allocated_nodes() > 2*new_root_data.visits as usize {
                    self.tree.compact();
                }
            }
            else{
                self.tree.clear();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{puct, negate_score};
    use super::*;

    #[test]
    fn test_action_creation() {
        let action = Action::new(42);
        assert_eq!(action.action(), 42);
    }

    #[test]
    fn test_action_equality() {
        let a1 = Action::new(10);
        let a2 = Action::new(10);
        let a3 = Action::new(11);

        assert_eq!(a1, a2, "Two actions with the same index should be equal");
        assert_ne!(a1, a3, "Actions with different indices should not be equal");
    }

    #[test]
    fn test_action_hashability() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        let a1 = Action::new(5);
        let a2 = Action::new(5);

        set.insert(a1);

        assert!(set.contains(&a2), "Action should be hashable and work in a HashSet");
        assert_eq!(set.len(), 1, "HashSet should handle identical Actions correctly");
    }

    fn dummy_select(_score: f32, _node_visits: i32, _parent_visits: i32, policy: f32, _c: f32) -> f32 {
        policy
    }

    fn identity_score(s: f32) -> f32 {
        s
    }

    const N: usize = 2;

    #[test]
    fn test_engine_initialization() {
        let engine = Engine::<N>::new();
        assert!(engine.tree.root().is_none());
    }

    #[test]
    fn test_select_on_empty_tree_returns_empty() {
        let engine = Engine::<N>::new();
        let mut path = Vec::new();
        let selection = engine.select(&mut path, &dummy_select, 1.0);

        assert_eq!(selection, SelectionResult::Empty);
        assert!(path.is_empty());
    }

    #[test]
    fn test_update_empty_creates_root() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let selection = engine.select(&mut path, &dummy_select, 1.0);

        let evaluation = StateEvaluation::Active(1.0, [0.7, 0.3]);
        let result = engine.update(evaluation, selection, negate_score);
        assert!(result.is_ok());

        let root_id = engine.tree.root().unwrap();
        let root = engine.tree.get(root_id).unwrap();
        assert_eq!(root.data().visits, 1);
        assert_eq!(root.data().state, MctsNodeState::Active);
    }

    #[test]
    fn test_select_active_node_and_expand_child() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let selection = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.7, 0.3]);
        engine.update(evaluation, selection, negate_score).unwrap();

        let selection2 = engine.select(&mut path, &dummy_select, 1.0);

        match selection2 {
            SelectionResult::Active(id, action) => {
                assert_eq!(id.0, engine.tree.root().unwrap());
                assert_eq!(action, Action(0));
            },
            _ => panic!("Expected Active selection"),
        }
        assert_eq!(path, vec![Action(0)]);

        let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
        let result = engine.update(evaluation, selection2, negate_score);
        assert!(result.is_ok());
    }

    #[test]
    fn test_backpropagate_values_correctly() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel_empty = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.7, 0.3]);
        engine.update(evaluation, sel_empty, negate_score).unwrap();

        let sel_active = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
        engine.update(evaluation, sel_active, negate_score).unwrap();

        let root_id = engine.tree.root().unwrap();
        let root = engine.tree.get(root_id).unwrap();

        assert_eq!(root.data().visits, 2);
        assert_eq!(root.data().children_visits[0], 1);
        assert_eq!(root.data().children_scores[0], 1.0);
    }

    #[test]
    fn test_terminal_node_selection_and_state() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel_empty = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Terminal(42.0);
        engine.update(evaluation, sel_empty, identity_score).unwrap();

        let root_id = engine.tree.root().unwrap();
        let root = engine.tree.get(root_id).unwrap();
        assert_eq!(root.data().state, MctsNodeState::Terminal(42.0));

        let sel_terminal = engine.select(&mut path, &dummy_select, 1.0);

        match sel_terminal {
            SelectionResult::Terminal(id, score) => {
                assert_eq!(id.0, root_id);
                assert_eq!(score, 42.0);
            },
            _ => panic!("Expected Terminal selection"),
        }
    }

    #[test]
    #[should_panic(expected = "Error during the selection of the best action.")]
    fn test_best_child_panics_if_no_legal_moves() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel_empty = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.0, 0.0]);
        engine.update(evaluation, sel_empty, negate_score).unwrap();

        engine.select(&mut path, &dummy_select, 1.0);
    }

    #[test]
    fn test_expand_on_terminal_returns_error() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel_empty = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Terminal(1.0);
        engine.update(evaluation, sel_empty, identity_score).unwrap();

        let sel_terminal = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
        let result = engine.expand(evaluation, sel_terminal);

        assert!(matches!(result, Err(MctsEngineError::SelectionIsTerminal)));
    }

    #[test]
    fn test_adding_existing_child_returns_error() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel_empty = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.7, 0.3]);
        engine.update(evaluation, sel_empty, negate_score).unwrap();

        let sel_active = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
        engine.update(evaluation, sel_active, negate_score).unwrap();

        let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
        let result = engine.update(evaluation, sel_active, negate_score);
        assert!(matches!(result, Err(MctsEngineError::ChildAlreadyExists)));
    }

    #[test]
    fn test_scores_empty_engine() {
        let engine = Engine::<N>::new();
        // Vérifie qu'un arbre vide renvoie bien un tableau de zéros
        assert_eq!(engine.scores(), [0; N]);
    }

    #[test]
    fn test_scores_after_root_expansion() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.5, 0.5]);
        engine.update(evaluation, sel, identity_score).unwrap();

        assert_eq!(engine.scores(), [0; N]);
    }

    #[test]
    fn test_scores_after_multiple_updates() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel_empty = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.7, 0.3]);
        engine.update(evaluation, sel_empty, identity_score).unwrap();

        for _ in 0..3 {
            let sel = engine.select(&mut path, &dummy_select, 1.0);
            let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
            engine.update(evaluation, sel, identity_score).unwrap();
        }

        let sel = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(1.0, [0.5, 0.5]);
        engine.update(evaluation, sel, identity_score).unwrap();

        let expected = [4, 0];
        assert_eq!(engine.scores(), expected);
    }

    #[test]
    #[allow(unused_assignments)]
    fn test_scores_are_independent_copies() {
        let mut engine = Engine::<N>::new();
        let mut path = Vec::new();

        let sel = engine.select(&mut path, &dummy_select, 1.0);
        let evaluation = StateEvaluation::Active(0.0, [0.5, 0.5]);
        engine.update(evaluation, sel, identity_score).unwrap();

        let mut scores = engine.scores();


        scores[0] = 999;

        assert_ne!(engine.scores()[0], 999);
        assert_eq!(engine.scores()[0], 0);
    }

    #[test]
    fn test_scnerario_1(){
        let c = std::f32::consts::SQRT_2;
        let mut path = Vec::new();
        let mut engine = Engine::<3>::new();

        let selection = engine.select(&mut path, &puct, c);
        let evaluation = StateEvaluation::Active(0.1, [0.5, 0.3, 0.2]);
        let node  = engine.expand(evaluation, selection).unwrap();
        engine.backpropagate(node, 0.1, negate_score).unwrap();

        assert_eq!(engine.scores(), [0, 0, 0]);
        assert_eq!(path.len(), 0);

        let selection = engine.select(&mut path, &puct, c);
        let evaluation = StateEvaluation::Active(0.5, [1., 0., 0.]);
        let node  = engine.expand(evaluation, selection).unwrap();
        engine.backpropagate(node, 0.5, negate_score).unwrap();

        assert_eq!(engine.scores(), [1, 0, 0]);
        assert_eq!(path.as_slice(), &[Action(0)]);

        let selection = engine.select(&mut path, &puct, c);
        let evaluation = StateEvaluation::Active(1., [1., 0., 0.]);
        let node  = engine.expand(evaluation, selection).unwrap();
        engine.backpropagate(node, 1.0, negate_score).unwrap();

        assert_eq!(engine.scores(), [2, 0, 0]);
        assert_eq!(path.as_slice(), &[Action(0), Action(0)]);

        let selection = engine.select(&mut path, &puct, c);
        let evaluation = StateEvaluation::Terminal(-0.9);
        let node  = engine.expand(evaluation, selection).unwrap();
        engine.backpropagate(node, -0.9, negate_score).unwrap();

        assert_eq!(engine.scores(), [2, 1, 0]);
        assert_eq!(path.as_slice(), &[Action(1)]);

        let selection = engine.select(&mut path, &puct, c);
        let evaluation = StateEvaluation::Terminal(-0.9);
        let node  = engine.expand(evaluation, selection).unwrap();
        engine.backpropagate(node, -0.9, negate_score).unwrap();

        assert_eq!(engine.scores(), [2, 1, 1]);
        assert_eq!(path.as_slice(), &[Action(2)]);
    }

    fn setup_engine() -> Engine<3> {
        let mut engine = Engine::<3>::new();
        let mut path = Vec::new();

        let sel = engine.select(&mut path, &|_,_,_,_,_| 0.0, 1.0);
        let evaluation = StateEvaluation::Active(0., [1., 0., 0.]);
        let node = engine.expand(evaluation, sel).unwrap();
        engine.backpropagate(node, 0.0, negate_score).unwrap();


        let sel = engine.select(&mut path, &|_,_,_,_,_| 0.0, 1.0);
        let evaluation = StateEvaluation::Active(1., [1., 0., 0.]);
        let node = engine.expand(evaluation, sel).unwrap();
        engine.backpropagate(node, 1.0, negate_score).unwrap();

        engine
    }

    #[test]
    fn test_commit_action_success() {
        let mut engine = setup_engine();

        engine.tree.root().unwrap();
        engine.commit_action(Action(0));

        let new_root = engine.tree.root().unwrap();

        assert_eq!(engine.tree.data(new_root).unwrap().visits, 1);
        assert_eq!(engine.tree.get(new_root).unwrap().data().action, Action(0));
    }

    #[test]
    fn test_commit_action_unexplored_leads_to_clear() {
        let mut engine = setup_engine();

        engine.commit_action(Action(2));
        assert!(engine.tree.root().is_none());
    }

    #[test]
    fn test_commit_action_no_root_does_nothing() {
        let mut engine = Engine::<3>::new();

        engine.commit_action(Action(0));

        assert!(engine.tree.root().is_none());
    }

    #[test]
    #[should_panic(expected = "index out of bounds")]
    fn test_commit_action_out_of_bounds_panics() {
        let mut engine = setup_engine();

        engine.commit_action(Action(5));
    }

    #[test]
    fn test_compact_trigger() {
        let mut engine = setup_engine();

        let root = engine.tree.root().unwrap();
        engine.tree.get_mut(root).unwrap().data_mut().visits = 100;

        let initial_nodes = engine.tree.allocated_nodes();

        engine.commit_action(Action(0));
        assert!(engine.tree.allocated_nodes() <= initial_nodes);
    }
}
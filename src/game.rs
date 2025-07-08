//! Module defining traits for games and game evaluators used in MCTS.

/// Trait defining the interface for a game that can be used with MCTS.
///
/// Implementations of this trait provide the core game logic,
/// allowing the MCTS algorithm to simulate and analyze game states.
///
/// # Type Parameters
/// - `N`: The number of possible actions in the game. This is a constant generic
///        parameter, meaning the number of actions is fixed at compile time.
pub trait Game<const N: usize>{
    /// The associated type representing the immutable state of the game.
    /// This type should ideally be lightweight. It can be use
    /// in `GameEvaluator` or AI model for make prediction.
    type State;

    /// Creates a new instance of the game in its initial state.
    ///
    /// This is typically the starting point for any new MCTS simulation.
    ///
    /// # Returns
    /// A new game instance initialized to its starting state.
    ///
    /// # Examples
    /// ```rust*
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::GameTest;
    /// let game = GameTest::new();
    /// // game is now in its initial state.
    /// ```
    fn new() -> Self;

    /// Returns an array indicating which actions are currently valid from the current game state.
    ///
    /// An action is valid if it can be played at this specific moment in the game.
    /// Actions are identified by their index (from 0 to `N-1`).
    ///
    /// # Returns
    /// An array of booleans where `true` at index `i` means the action `i` is valid,
    /// and `false` means it's invalid.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::*;
    /// let game = GameTest::new();
    /// let actions = game.get_actions();
    /// // For GameTest, initially all actions are valid: [true, true, true, true]
    /// assert_eq!(actions, [true, true, true, true]);
    /// ```
    fn get_actions(&self) -> [bool; N];

    /// Determines if the game has reached a terminal state (i.e., it's over).
    ///
    /// A game is finished if no more moves can be made, or if a win/loss/draw
    /// condition has been met.
    ///
    /// # Returns
    /// `true` if the game is finished, `false` otherwise.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::GameTest;
    /// let mut game = GameTest::new();
    /// assert!(!game.is_finish());
    /// game.play(0); game.play(1); game.play(2); game.play(3);
    /// assert!(game.is_finish());
    /// ```
    fn is_finish(&self) -> bool;

    /// Applies a given action to the game, transitioning it to a new state.
    ///
    /// This method modifies the current game instance. It's assumed that the
    /// `action` provided is valid according to `get_actions`.
    ///
    /// # Parameters
    /// - `action`: The index of the action to be played.
    ///
    /// # Panics
    /// This method typically assumes `action` is valid. If `action` is out of bounds
    /// or invalid for the current state, the behavior is implementation-defined
    /// and may lead to a panic or incorrect state.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::GameTest;
    /// let mut game = GameTest::new();
    /// assert_eq!(game.get_state(), [-1, -1, -1, -1]);
    /// game.play(0);
    /// assert_eq!(game.get_state(), [0, -1, -1, -1]);
    /// ```
    fn play(&mut self, action: usize);

    /// Returns an immutable representation of the current game state.
    ///
    /// This state is typically used by `GameEvaluator` to assess the game
    /// without needing to clone the entire `Game` instance.
    ///
    /// # Returns
    /// The current state of the game.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::GameTest;
    /// let mut game = GameTest::new();
    /// let initial_state = game.get_state();
    /// // State for GameTest is an array [i32; 4] representing moves made.
    /// assert_eq!(initial_state, [-1, -1, -1, -1]);
    /// ```
    fn get_state(&self) -> Self::State;

    /// Returns the final result of the game if it has finished.
    ///
    /// The result is typically a value representing the outcome from the perspective
    /// of the player whose turn it was when the game finished.
    /// Common values include:
    /// - `1.0`: Win for the current player.
    /// - `0.0`: Draw.
    /// - `-1.0`: Loss for the current player / win for opponent.
    ///
    /// # Returns
    /// An `Option<f64>`:
    /// - `Some(value)` if the game is finished and a result can be determined.
    /// - `None` if the game is not yet finished.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::GameTest;
    /// let mut game = GameTest::new();
    /// assert_eq!(game.get_result(), None);
    /// game.play(0); game.play(1); game.play(2); game.play(3); // Finish the game
    /// // The exact result depends on GameTest's internal logic.
    /// // assert_eq!(game.get_result(), Some(...));
    /// ```
    fn get_result(&self) -> Option<f64>;
    
    /// Creates a deep copy of the current game instance.
    ///
    /// This is crucial for MCTS as simulations often require creating independent
    /// branches from a given game state without affecting the original.
    ///
    /// # Returns
    /// A new, independent instance of the game with the same state.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::Game;
    /// use simple_mcts::test_utils::*;
    /// let original_game = GameTest::new();
    /// let cloned_game = original_game.clone();
    /// assert_eq!(original_game.get_state(), cloned_game.get_state());
    /// ```
    fn clone(&self) -> Self;
}
/// Trait for evaluating game states and providing policy suggestions.
///
/// Implementations of this trait are typically used by the MCTS algorithm
/// during the simulation and expansion phases to assess game states and
/// determine probabilities for subsequent actions. This trait allows for
/// plugging in external evaluation models (e.g., neural networks).
///
/// # Type Parameters
/// - `T`: The game type that this evaluator is designed for, implementing the `Game` trait.
/// - `N`: The number of possible actions in the game, a constant generic parameter.

pub trait GameEvaluator<T: Game<N>, const N: usize>{
    /// Evaluates a given game state and returns an estimated value and
    /// a probability distribution over possible actions (policy).
    ///
    /// The value estimate typically represents the likelihood of winning from this state,
    /// often from the perspective of the current player, and commonly in the range `[-1.0, 1.0]`.
    /// The policy array gives probabilities for each action, where `policy[i]` is the probability
    /// of taking action `i`. The sum of probabilities for valid actions should ideally be 1.0.
    ///
    /// # Parameters
    /// - `state`: The game state to evaluate. This is typically a `Game::State` type.
    ///
    /// # Returns
    /// A tuple containing:
    /// - `f64`: The estimated value of the state (e.g., from the current player's perspective).
    /// - `[f64; N]`: An array of action probabilities (policy), where each element corresponds
    ///   to the probability of taking that action from the given state.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::{test_utils::{GameTest, GameEvaluatorTest2}, Game, GameEvaluator};
    /// let evaluator = GameEvaluatorTest2::new();
    /// let game = GameTest::new();
    /// let initial_state = game.get_state();
    /// let (value, policy) = evaluator.evaluate(initial_state);
    /// println!("Initial state value: {}, policy: {:?}", value, policy);
    /// ```
    fn evaluate(&self, state: T::State) -> (f64, [f64; N]);
}
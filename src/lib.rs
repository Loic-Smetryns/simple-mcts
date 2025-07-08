//! A Rust library providing a highly configurable and efficient
//! Monte Carlo Tree Search (MCTS) implementation.
//!
//! This library supports various game types and integrates with external
//! game evaluators for flexible AI development. It includes both
//! single-instance and batch-processing capabilities for MCTS.
//!
//! # Modules
//! - `tree`: Implements the core tree data structure used by MCTS.
//! - `game`: Defines traits for game logic and state evaluation.
//! - `mcts`: Provides the single-instance MCTS algorithm.
//! - `mcts_batch`: Offers an MCTS implementation capable of processing multiple
//!                 game instances in parallel for increased throughput.
//! - `utils`: Contains general utility functions.
//! - `test_utils`: Provides helper implementations for testing the MCTS algorithms.
//!
//! # Examples
//! ```rust
//! use simple_mcts::{Mcts, test_utils::{GameTest, GameEvaluatorTest2}, MctsError};
//!
//! fn main() -> Result<(), MctsError> {
//!     let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
//!     let evaluator = GameEvaluatorTest2::new();
//!
//!     // Perform 100 MCTS iterations
//!     for _ in 0..100 {
//!         mcts.iterate(&evaluator)?;
//!     }
//!
//!     // Get the best action based on visit counts
//!     let (score, policy) = mcts.get_result();
//!     println!("Best action score: {}, Policy: {:?}", score, policy);
//!
//!     // Play the best action and update the MCTS tree
//!     let best_action_index = policy.iter()
//!                                   .enumerate()
//!                                   .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
//!                                   .map(|(index, _)| index)
//!                                   .unwrap_or(0); // Default to first action if policy is empty
//!     mcts.play(best_action_index)?;
//!
//!     // Continue with the next game state
//!     Ok(())
//! }
//! ```
//!
//! This library aims to be a robust foundation for AI development in board games,
//! particularly those benefiting from tree search algorithms like AlphaZero.

mod tree;
mod game;
mod mcts;
mod mcts_batch;
pub mod utils;

#[doc(hidden)]
pub mod test_utils;

use tree::*;
pub use game::*;
pub use mcts::*;
pub use mcts_batch::*;
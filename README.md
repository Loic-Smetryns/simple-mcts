# simple-mcts: A Simple and Configurable Monte Carlo Tree Search (MCTS) Library for Rust

`simple-mcts` is a Rust library providing a straightforward and configurable implementation of the Monte Carlo Tree Search (MCTS) algorithm. It's designed for easy integration into various game AI projects, supporting both single-instance and batch-processing MCTS simulations.

This library aims to provide a functional foundation for AI development in board games, particularly those benefiting from tree search algorithms.

## Features

    * Generic Game Interface: Define your game logic using simple traits (Game and GameEvaluator).

    * Configurable MCTS: Adjust MCTS parameters to fine-tune search behavior.

    * Batch Processing: Run multiple MCTS simulations efficiently, which can be useful for experimentation or high-throughput scenarios.

    * Error Handling: Clear error types for better debugging.

    * Clear Modularity: Organized code structure for better understanding.

## Getting Started

Add `simple-mcts` to your `Cargo.toml`:

```toml
[dependencies]
simple-mcts = "0.1.0" # Check Crates.io for the latest version
rand = "0.9.1"
```

## Basic Usage Example

Here's a quick example demonstrating how to use the Mcts algorithm with a simple test game.

```rust
use simple_mcts::{Mcts, test_utils::{GameTest, GameEvaluatorTest2}, MctsError};

fn main() -> Result<(), MctsError> {
    // Initialize a new MCTS instance for GameTest (a simple 4-action game)
    let mut mcts: Mcts<GameTest, 4> = Mcts::<GameTest, 4>::new();
    // Initialize a game evaluator (e.g., a machine learning model or a heuristic)
    let evaluator = GameEvaluatorTest2::new(); // test_utils provides a dummy evaluator

    // Perform MCTS iterations to build the search tree
    // Each iteration involves selection, expansion, simulation, and backpropagation
    for _ in 0..100 { // Perform 100 search iterations
        mcts.iterate(&evaluator)?;
    }

    // Get the best action based on the MCTS search results (e.g., most visited action)
    let (score, policy) = mcts.get_result();
    println!("Best action score: {}, Policy: {:?}", score, policy);

    // Simulate playing the best action in the game and update the MCTS tree
    let best_action_index = policy.iter()
                                  .enumerate()
                                  .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                                  .map(|(index, _)| index)
                                  .unwrap_or(0); // Defaults to the first action if policy is empty
    mcts.play(best_action_index)?;

    println!("Game state after playing action {}", best_action_index);
    // You can now continue the MCTS search from the new game state
    // mcts.iterate(&evaluator)?;

    Ok(())
}
```

## Implementing Your Own Game

To use `simple-mcts` with your own game, you need to implement the `Game` and `GameEvaluator` traits:

    * `Game<const N: usize>`: Defines the game's rules, state representation, valid actions, and how to play a move.

    * `GameEvaluator<T: Game<N>, const N: usize>`: Provides a way to evaluate a game state, returning an estimated value (e.g., win probability) and a policy (probabilities of taking each action). This is typically where you would integrate a machine learning model or a heuristic.

See the `game.rs` and `test_utils.rs` files for detailed examples of these trait implementations.

## Modules

* `tree`: Implements the core tree data structure used by MCTS.

* `game`: Defines traits for game logic and state evaluation (`Game`, `GameEvaluator`).

* `mcts`: Provides the single-instance MCTS algorithm.

* `mcts_batch`: Offers an MCTS implementation capable of processing multiple game instances in parallel.

* `utils`: Contains utility functions.

* `test_utils`: Provides helper implementations for testing the MCTS algorithms.

## License

This project is licensed under the MIT License. See the LICENSE file for details.
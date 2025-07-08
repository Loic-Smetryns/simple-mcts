//! Test utilities for MCTS implementation

use crate::{Game, GameEvaluator};

/// A simple test game implementation
pub struct GameTest {
    state: Vec<i32>,
}

impl Game<4> for GameTest {
    type State = [i32; 4];

    fn new() -> Self {
        GameTest {
            state: Vec::<i32>::new(),
        }
    }

    fn get_actions(&self) -> [bool; 4] {
        std::array::from_fn(|index| !self.state.contains(&(index as i32)))
    }

    fn is_finish(&self) -> bool{
        self.state.len() == 4
    }

    fn play(&mut self, action: usize) {
        self.state.push(action as i32);
    }

    fn get_state(&self) -> Self::State {
        std::array::from_fn(|index| {
            if index < self.state.len() {
                self.state[index]
            } else {
                -1
            }
        })
    }

    fn get_result(&self) -> Option<f64> {
        if self.state.len() != 4 {
            None
        } else {
            let rel = (self.state[0] + self.state[2]) - (self.state[1] + self.state[3]);

            if rel == 0 {
                Some(0.0)
            } else if rel > 0 {
                Some(1.0)
            } else {
                Some(-1.0)
            }
        }
    }

    fn clone(&self) -> Self {
        GameTest {
            state: self.state.clone(),
        }
    }
}

/// A simple test evaluator implementation
pub struct GameEvaluatorTest;

impl GameEvaluatorTest {
    #[allow(dead_code)]
    pub fn new() -> Self {
        GameEvaluatorTest {}
    }
}

impl GameEvaluator<GameTest, 4> for GameEvaluatorTest {
    fn evaluate(&self, state: <GameTest as Game<4>>::State) -> (f64, [f64; 4]) {
        let last = state.into_iter().filter(|i| *i != -1).last();

        if let Some(last) = last {
            match last {
                0 => (-0.2, [0.41, 0.39, 0.09, 0.11]),
                1 => (0.2, [0.09, 0.11, 0.41, 0.39]),
                2 => (-0.9, [0.1, 0.12, 0.13, 0.65]),
                3 => (0.9, [0.13, 0.65, 0.1, 0.12]),
                _ => (0.0, [0.25, 0.25, 0.25, 0.25]),
            }
        } else {
            (0.0, [0.2, 0.7, 0.06, 0.04])
        }
    }
}

/// Another test evaluator implementation with different behavior
pub struct GameEvaluatorTest2;

impl GameEvaluatorTest2 {
    #[allow(dead_code)]
    pub fn new() -> Self {
        GameEvaluatorTest2 {}
    }
}

impl GameEvaluator<GameTest, 4> for GameEvaluatorTest2 {
    fn evaluate(&self, state: <GameTest as Game<4>>::State) -> (f64, [f64; 4]) {
        let last = state.into_iter().filter(|i| *i != -1).last();

        if let Some(last) = last {
            match last {
                0 => (0.5, [0.1, 0.15, 0.25, 0.5]),
                1 => (0.25, [0.1, 0.15, 0.25, 0.5]),
                2 => (-0.25, [0.1, 0.15, 0.25, 0.5]),
                3 => (-0.5, [0.1, 0.15, 0.25, 0.5]),
                _ => (0.0, [0.25, 0.25, 0.25, 0.25]),
            }
        } else {
            (0.0, [0.1, 0.15, 0.25, 0.5])
        }
    }
}

/// Utility function to compare float arrays with tolerance
///
/// # Parameters
/// - `a`: First array
/// - `b`: Second array
///
/// # Returns
/// `true` if all elements are approximately equal
#[allow(dead_code)]
pub fn compare_array<const N: usize>(a: &[f64; N], b: &[f64; N]) -> bool{
    (0..N).all(|index|{
        (a[index]-b[index]).abs() < 1e-8
    })
}
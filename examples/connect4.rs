use simple_mcts::{negate_score, puct, visits_to_probabilities, Action, Engine, StateEvaluation};
use rand::seq::IndexedRandom;

#[derive(Copy, Clone, Debug, PartialEq)]
enum Tile {
    Empty, Red, Yellow,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Player {
    Red, Yellow,
}

#[derive(Copy, Clone, Debug)]
struct Connect4 {
    board: [[Tile; 7]; 6],
    player: Player,
}

impl Connect4 {
    pub fn new() -> Self {
        Connect4 {
            board: [[Tile::Empty; 7]; 6],
            player: Player::Red,
        }
    }

    pub fn actions_mask(&self) -> [bool; 7] {
        std::array::from_fn(|x| self.board[0][x] == Tile::Empty)
    }

    pub fn play(&mut self, action: Action) {
        let x = action.action();
        assert!(x < 7, "invalid action");

        for y in (0..6).rev() {
            if self.board[y][x] == Tile::Empty {
                self.board[y][x] = if self.player == Player::Red { Tile::Red } else { Tile::Yellow };
                self.player = if self.player == Player::Red { Player::Yellow } else { Player::Red };
                return;
            }
        }
        panic!("Tried to play in a full column!");
    }

    pub fn winner(&self) -> Option<Tile> {
        for y in 0..6 {
            for x in 0..4 {
                if self.board[y][x] != Tile::Empty &&
                    self.board[y][x] == self.board[y][x+1] &&
                    self.board[y][x] == self.board[y][x+2] &&
                    self.board[y][x] == self.board[y][x+3] {
                    return Some(self.board[y][x]);
                }
            }
        }

        for x in 0..7 {
            for y in 0..3 {
                if self.board[y][x] != Tile::Empty &&
                    self.board[y][x] == self.board[y+1][x] &&
                    self.board[y][x] == self.board[y+2][x] &&
                    self.board[y][x] == self.board[y+3][x] {
                    return Some(self.board[y][x]);
                }
            }
        }

        for y in 0..3 {
            for x in 0..4 {
                if self.board[y][x] != Tile::Empty &&
                    self.board[y][x] == self.board[y+1][x+1] &&
                    self.board[y][x] == self.board[y+2][x+2] &&
                    self.board[y][x] == self.board[y+3][x+3] {
                    return Some(self.board[y][x]);
                }
            }
        }

        for y in 3..6 {
            for x in 0..4 {
                if self.board[y][x] != Tile::Empty &&
                    self.board[y][x] == self.board[y-1][x+1] &&
                    self.board[y][x] == self.board[y-2][x+2] &&
                    self.board[y][x] == self.board[y-3][x+3] {
                    return Some(self.board[y][x]);
                }
            }
        }
        None
    }

    pub fn is_draw(&self) -> bool {
        self.board[0].iter().all(|&tile| tile != Tile::Empty)
    }

    pub fn score(&self) -> Option<f32> {
        if let Some(winner) = self.winner() {
            match winner {
                Tile::Red => Some(if self.player == Player::Red { 1.0 } else { -1.0 }),
                Tile::Yellow => Some(if self.player == Player::Yellow { 1.0 } else { -1.0 }),
                Tile::Empty => unreachable!(),
            }
        } else if self.is_draw() {
            Some(0.0)
        } else {
            None
        }
    }
}

fn rollout(state: &Connect4) -> f32 {
    let mut current = state.clone();
    let mut rng = rand::rng();

    loop {
        if let Some(winner) = current.winner() {
            return match winner {
                Tile::Red => if state.player == Player::Red { 1.0 } else { -1.0 },
                Tile::Yellow => if state.player == Player::Yellow { 1.0 } else { -1.0 },
                Tile::Empty => unreachable!(),
            };
        } else if current.is_draw() {
            return 0.0;
        }

        let mask = current.actions_mask();
        let valid_actions: Vec<usize> = mask.iter().enumerate()
            .filter(|&(_, &m)| m)
            .map(|(i, _)| i)
            .collect();

        let random_action = valid_actions.choose(&mut rng).unwrap();
        current.play(Action::new(*random_action));
    }
}

fn evaluation(state: &Connect4) -> StateEvaluation<7> {
    if let Some(score) = state.score() {
        StateEvaluation::Terminal(-score)
    } else {
        let mask = state.actions_mask();
        let total = mask.iter().filter(|&&b| b).count() as f32;

        let policies = std::array::from_fn(|i| if mask[i] { 1.0 / total } else { 0.0 });
        let rollout_value = rollout(state);

        StateEvaluation::Active(-rollout_value, policies)
    }
}

fn main() {
    const C: f32 = std::f32::consts::SQRT_2;

    let mut mcts = Engine::<7>::new();
    let game = Connect4::new();
    let mut path = Vec::new();

    for _ in 0..100_000 {
        let selection = mcts.select(&mut path, &puct, C);

        let mut temp_game = game.clone();
        for action in path.iter() {
            temp_game.play(*action);
        }

        mcts.update(evaluation(&temp_game), selection, negate_score)
            .expect("Error while updating mcts");
    }


    let scores = visits_to_probabilities(mcts.scores());

    println!("| {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} | {:>5.2} |",
             scores[0], scores[1], scores[2], scores[3], scores[4], scores[5], scores[6]);
    println!("   Col0    Col1    Col2    Col3    Col4    Col5    Col6");
}
use simple_mcts::{negate_score, puct, visits_to_probabilities, Action, Engine, MctsBatch, MctsConfig, StateEvaluation};

#[derive(Copy, Clone, Debug, PartialEq)]
enum Tile{
    Empty, Cross, Circle,
}

#[derive(Copy, Clone, Debug, PartialEq)]
enum Player{
    Cross, Circle,
}

#[derive(Copy, Clone, Debug)]
struct TicTacToe {
    board: [[Tile; 3]; 3],
    player: Player,
}

impl TicTacToe {
    pub fn new() -> Self {
        TicTacToe {
            board: [[Tile::Empty; 3]; 3],
            player: Player::Cross,
        }
    }

    pub fn actions_mask(&self) -> [bool; 9] {
        std::array::from_fn(|i|{
            let x = i % 3;
            let y = i / 3;

            match self.board[y][x] {
                Tile::Empty => true,
                _ => false,
            }
        })
    }

    pub fn play(&mut self, action: Action){
        let action = action.action();
        assert!(action < 9, "invalid action");


        let x = action % 3;
        let y = action / 3;

        self.board[y][x] = if self.player == Player::Cross { Tile::Cross } else { Tile::Circle };
        self.player = if self.player == Player::Cross { Player::Circle } else { Player::Cross };
    }

    pub fn score(&self) -> Option<f32>{
        let mut winner: Option<Tile> = None;

        for i in (0..3) {
            if self.board[i][0] != Tile::Empty && self.board[i][0] == self.board[i][1] && self.board[i][1] == self.board[i][2] ||
                self.board[0][i] != Tile::Empty && self.board[0][i] == self.board[1][i] && self.board[1][i] == self.board[2][i] {
                winner = Some(self.board[i][i]);
            }
        }

        if self.board[0][0] != Tile::Empty && self.board[0][0] == self.board[1][1] && self.board[1][1] == self.board[2][2] ||
            self.board[0][2] != Tile::Empty && self.board[0][2] == self.board[1][1] && self.board[1][1] == self.board[2][0] {
            winner = Some(self.board[1][1]);
        }

        match winner {
            Some(Tile::Cross) => { Some(if self.player == Player::Cross { 1.} else { -1. }) },
            Some(Tile::Circle) => { Some(if self.player == Player::Cross { -1.} else { 1. }) },
            _ => {
                if self.board.iter().flatten().all(|tile| *tile!=Tile::Empty) { Some(0.0) } else { None }
            },
        }
    }
}

fn evaluation(state: &TicTacToe) -> StateEvaluation<9>{
    if let Some(score) = state.score() {
        StateEvaluation::Terminal(-score)
    }
    else {
        let mask = state.actions_mask();
        let total = mask.iter().filter(|&&b| b).count() as f32;

        StateEvaluation::Active(0., std::array::from_fn(|i|  if mask[i] { 1./total } else { 0. }))
    }
}

fn main(){
    const C: f32 = std::f32::consts::SQRT_2;

    let mut mcts = Engine::<9>::new();
    let mut game = TicTacToe::new();
    let mut path = Vec::new();

    for _ in 0..100000 {
        let selection = mcts.select(&mut path, &puct, C);

        let mut temp_game = game.clone();
        for action in path.iter_mut() {
            temp_game.play(*action);
        }

        mcts.update(evaluation(&temp_game), selection, negate_score).expect("Error while updating mcts");
    }

    let scores = visits_to_probabilities(mcts.scores());

    println!("| {:.2}  {:.2}  {:.2} |", scores[0], scores[1], scores[2]);
    println!("| {:.2}  {:.2}  {:.2} |", scores[3], scores[4], scores[5]);
    println!("| {:.2}  {:.2}  {:.2} |", scores[6], scores[7], scores[8]);
}
//! Batch processing implementation for Monte Carlo Tree Search (MCTS) algorithm
//!
//! Provides parallel simulation capabilities for multiple MCTS instances

use std::{sync::atomic::{AtomicU8, Ordering}, time::{SystemTime, UNIX_EPOCH}};

use rand::{rngs::StdRng, SeedableRng};

use crate::{utils, Game, GameEvaluator, Mcts, MctsConfig, MctsError, MctsState, SelectionFunction};

/// Configuration parameters for an `MctsBatch` manager.
///
/// This struct extends `MctsConfig` with batch-specific settings like the random
/// number generator seed for reproducible batch simulations.
///
/// # Type Parameters
/// - `N`: The number of possible actions in the game.
pub struct MctsBatchConfig<const N: usize>{
    /// The exploration coefficient for MCTS instances within the batch.
    pub exploration_coef: f64,
    /// The selection function for MCTS instances within the batch.
    pub selection_function: SelectionFunction<N>,
    // An optional seed for the random number generator used in the batch.
    ///
    /// Providing a `Some(value)` will initialize the RNG with a fixed seed,
    /// ensuring reproducible simulation results across runs. If `None`,
    /// a new seed based on the current time will be used, leading to
    /// non-reproducible runs (but more "random" behavior).
    pub seed: Option<u64>,
}

impl<const N: usize> MctsBatchConfig<N>{
    /// The default configuration for an `MctsBatch`.
    ///
    /// It inherits the default exploration coefficient and selection function from `MctsConfig::DEFAULT`,
    /// and uses `None` for the seed, meaning simulations will not be reproducible by default unless a seed is provided.
    pub const DEFAULT: MctsBatchConfig<N> = MctsBatchConfig::<N>{
        exploration_coef: MctsConfig::<N>::DEFAULT.exploration_coef,
        selection_function: MctsConfig::<N>::DEFAULT.selection_function,
        seed: None
    };
}

/// Represents the **history of a single MCTS instance**, capturing key data at each step:
/// - The **game state** at that point.
/// - The **estimated value** of the state.
/// - The **action probabilities** derived from the MCTS search.
#[allow(type_alias_bounds)]
type History<T: Game<N>, const N: usize> = Vec<(T, f64, [f64; N])>;

/// Manages a collection of **MCTS simulations** that can be processed in parallel.
///
/// `MctsBatch` tracks the complete history of each simulation, which is valuable for
/// training machine learning models or analyzing game outcomes.
///
/// # Type Parameters
/// - `T`: The type of game being simulated.
/// - `N`: The number of possible actions in the game.
pub struct MctsBatch<T: Game<N>, const N: usize>{
    /// Collection of MCTS instances with their histories
    instances: Vec<Option<(Mcts<T, N>, History<T, N>)>>,
    /// Number of active instances
    count: usize,
    /// Shared random number generator
    rand: StdRng,
    /// Current state of the batch processor
    state: MctsState,
    /// Config used for Mcts
    config: MctsConfig<N>
}

impl<T: Game<N>, const N: usize> MctsBatch<T, N>{
    /// Creates a new empty MCTS batch processor with the default configuration.
    ///
    /// The random number generator will be seeded based on the current system time.
    #[inline]
    pub fn new() -> Self{
        MctsBatch::from_config(&MctsBatchConfig::<N>::DEFAULT)
    }

    /// Creates a new empty MCTS batch processor from a specified configuration.
    ///
    /// This allows customizing the MCTS parameters for all instances in the batch,
    /// including the random number generator's seed for reproducibility.
    ///
    /// # Parameters
    /// - `config`: The `MctsBatchConfig` to use for this batch manager.
    #[inline]
    pub fn from_config(config: &MctsBatchConfig<N>) -> Self{
        MctsBatch {
            instances: Vec::new(),
            count: 0,
            rand: SeedableRng::seed_from_u64(
                if let Some(seed) = &config.seed { 
                    *seed 
                } else { 
                    (SystemTime::now().duration_since(UNIX_EPOCH).expect("").as_nanos()%u64::MAX as u128) as u64 
                }
            ),
            state: MctsState(AtomicU8::new(MctsState::USABLE)),
            config: MctsConfig { exploration_coef: config.exploration_coef, selection_function: config.selection_function }
        }
    }

    /// Returns the number of active MCTS instances currently managed by the batch processor.
    ///
    /// This count represents the simulations that are still ongoing and have not yet
    /// reached a terminal state or been otherwise removed from the batch.
    ///
    /// # Returns
    /// The `usize` representing the total number of active MCTS instances.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::{MctsBatch, test_utils::GameTest, MctsError, Game};
    ///
    /// fn main() -> Result<(), MctsError> {
    ///     let mut manager = MctsBatch::<GameTest, 4>::new();
    ///     manager.populate_from_game(vec![GameTest::new(), GameTest::new()])?;
    ///     
    ///     // Initially, the count should reflect the number of populated instances.
    ///     assert_eq!(manager.get_count(), 2);
    ///
    ///     // After some iterations, if games finish or are processed, the count might change.
    ///     // (Assuming `next` or other operations might reduce the count)
    ///     // manager.iterate(&evaluator)?; // If an iteration causes a game to finish
    ///     // manager.next()?; // If retrieving results removes finished games
    ///     
    ///     // Example: If one game finishes and is removed
    ///     // assert_eq!(manager.get_count(), 1); 
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn get_count(&self) -> usize{
        self.count
    }

    /// Returns the current operational state of the `MctsBatch` processor.
    ///
    /// This indicates whether the batch is `Usable` (ready for operations),
    /// `AwaitingSimulation` (waiting for external evaluation results), or `Locked`
    /// (temporarily busy with internal processing).
    ///
    /// # Returns
    /// A clone of the current `MctsState` of the batch processor.
    ///
    /// # Examples
    /// ```rust
    /// use simple_mcts::{MctsBatch, test_utils::GameTest, MctsError, MctsState, Game};
    ///
    /// fn main() -> Result<(), MctsError> {
    ///     let manager = MctsBatch::<GameTest, 4>::new();
    ///     // A newly created MctsBatch is typically in the Usable state.
    ///     assert_eq!(manager.get_state(), MctsState::USABLE);
    ///     
    ///     // The state would change, for example, after calling `start_iteration`
    ///     // (if MctsBatch had such a method that changed its state to AwaitingSimulation)
    ///     // or during internal processing.
    ///     Ok(())
    /// }
    /// ```
    #[inline]
    pub fn get_state(&self) -> u8{
        self.state.0.load(Ordering::SeqCst)
    }

    /// Populates the batch with new MCTS instances.
    ///
    /// This method requires the batch processor to be in a `Usable` state.
    ///
    /// # Parameters
    /// - `n`: The number of new instances to add.
    ///
    /// # Returns
    /// `Ok(())` if the batch is successfully populated.
    /// `Err(MctsError::InvalidState(_))` if the batch processor is not in the `Usable` state.
    pub fn populate(&mut self, mut n: usize) -> Result<(), MctsError>{
        match self.state.0.compare_exchange(MctsState::USABLE, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        let mut empty_place: usize = self.instances.len() - self.get_count();
        self.count += n;

        if empty_place > 0 && n > 0{
            for opt in &mut self.instances{
                if opt.is_none(){
                    *opt = Some((Mcts::<T, N>::new(), Vec::new()));
                    empty_place -= 1;
                    n -= 1;
                }

                if n == 0 { return Ok(()); }
                if empty_place == 0{ break; }
            }
        }

        for _ in 0..n{
            self.instances.push(Some((Mcts::<T, N>::from_config(&self.config), Vec::new())));
        }

        self.state.0.store(MctsState::USABLE, Ordering::Relaxed);
        Ok(())
    }

    /// **Populates the batch with new MCTS instances, each initialized with an existing game state.**
    ///
    /// This method requires the batch processor to be in a `Usable` state.
    ///
    /// # Parameters
    /// - `games`: A vector of game states to initialize the new instances.
    ///
    /// # Returns
    /// - `Ok(())` if the batch is successfully populated.
    /// - `Err(MctsError::InvalidState(_))` if the batch is not in the `Usable` state.
    pub fn populate_from_game(&mut self, mut games: Vec<T>) -> Result<(), MctsError>{
        match self.state.0.compare_exchange(MctsState::USABLE, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        let mut n: usize = games.len();
        let mut empty_place: usize = self.instances.len() - self.get_count();

        self.count += n;

        if empty_place > 0 && n > 0{
            for opt in &mut self.instances{
                if opt.is_none(){
                    *opt = Some((Mcts::<T, N>::from_game_with_config(games.pop().unwrap(), &self.config), Vec::new()));
                    empty_place -= 1;
                    n -= 1;
                }

                if n == 0 { return Ok(()); }
                if empty_place == 0{ break; }
            }
        }

        for _ in 0..n{
            self.instances.push(Some((Mcts::<T, N>::from_game(games.pop().unwrap()), Vec::new())));
        }

        self.state.0.store(MctsState::USABLE, Ordering::Relaxed);
        Ok(())
    }

    /// **Clears all instances from the batch**, resetting it to an empty state.
    ///
    /// This method requires the batch processor to be in a `Usable` state.
    ///
    /// # Returns
    /// - `Ok(())` if the batch is successfully cleared.
    /// - `Err(MctsError::InvalidState(_))` if the batch is not in the `Usable` state.
    pub fn clear(&mut self) -> Result<(), MctsError>{
        match self.state.0.compare_exchange(MctsState::USABLE, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        self.instances.clear();
        self.count = 0;

        self.state.0.store(MctsState::USABLE, Ordering::Relaxed);
        Ok(())
    }

    /// Performs **one full MCTS iteration** on all active instances in the batch.
    ///
    /// This method requires the batch processor to be in a `Usable` state.
    ///
    /// # Parameters
    /// - `evaluator`: The policy/value evaluator used for simulations.
    ///
    /// # Returns
    /// - `Ok(())` if all active instances complete an iteration successfully.
    /// - `Err(MctsError::InvalidState(_))`: If the batch is not in the `Usable` state.
    /// - Propagated errors from MCTS instances, such as:
    ///     - `MctsError::SearchAlreadyOver`
    ///     - `MctsError::InvalidEvaluationCount`
    ///     - `MctsError::ActionOutOfRange`
    ///     - `MctsError::InvalidAction`
    ///     - `MctsError::UnexploredAction`
    pub fn iterate(&mut self, evaluator: &dyn GameEvaluator<T, N>) -> Result<(), MctsError>{
        match self.state.0.compare_exchange(MctsState::USABLE, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        for opt in &mut self.instances{
            if let Some((mcts, _history)) = opt{
                if !mcts.get_game().is_finish() {
                    mcts.iterate(evaluator)?;
                }
            }
        }
        
        self.state.0.store(MctsState::USABLE, Ordering::Relaxed);
        Ok(())
    }

    /// Initiates the selection and expansion phases for all active MCTS instances in the batch.
    ///
    /// This method requires the batch processor to be in a `Usable` state
    /// and transitions it to `AwaitingSimulation`.
    ///
    /// # Returns
    /// `Ok(Vec<T::State>)` containing the game states from each instance that require external evaluation.
    /// `Err(MctsError::InvalidState(_))` if the batch processor is not in the `Usable` state.
    /// `Err(MctsError::SearchAlreadyOver)` if an MCTS instance's root node indicates the game is already finished.
    pub fn start_iteration(&mut self) -> Result<Vec<T::State>, MctsError>{
        match self.state.0.compare_exchange(MctsState::USABLE, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        let mut game_states: Vec<T::State> = Vec::with_capacity(self.get_count());

        for opt in &mut self.instances{
            if let Some((mcts, _history)) = opt{
                if !mcts.get_game().is_finish() {
                    game_states.push(mcts.start_iteration()?);
                }
            }
        }

        self.state.0.store(MctsState::AWAITING_SIMULATION, Ordering::Relaxed);
        Ok(game_states)
    }

    /// Applies the results of external simulations and performs backpropagation for all active MCTS instances.
    ///
    /// This method requires the batch processor to be in the `AwaitingSimulation` state
    /// and transitions it back to `Usable`.
    ///
    /// # Parameters
    /// - `evaluations`: A vector of tuples, where each tuple contains the estimated value (f64)
    ///                 and action probabilities ([f64; N]) for an MCTS instance.
    ///                 The order of evaluations should correspond to the order of game states
    ///                 returned by `start_iteration`.
    ///
    /// # Returns
    /// `Ok(())` if all simulations are successfully applied and backpropagation completes.
    /// `Err(MctsError::InvalidState(_))` if the batch processor is not in the `AwaitingSimulation` state.
    /// `Err(MctsError::InvalidEvaluationCount(expected, received))` if the number of provided
    ///                                  evaluations does not match the number of active instances
    ///                                  that were awaiting simulation.
    pub fn apply_simulation(&mut self, mut evaluations : Vec<(f64, [f64; N])>) -> Result<(), MctsError>{
        match self.state.0.compare_exchange(MctsState::AWAITING_SIMULATION, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        if evaluations.len() != self.get_count() {
            self.state.0.store(MctsState::AWAITING_SIMULATION, Ordering::Relaxed);
            return Err(MctsError::InvalidEvaluationCount(self.get_count(), evaluations.len()));
        }

        for opt in &mut self.instances.iter_mut().rev(){
            if let Some((mcts, history)) = opt{
                if mcts.get_game().is_finish() { continue; }
                let game = mcts.get_game().clone();

                mcts.apply_simulation(evaluations.pop().unwrap())?;
                let (value, policy) = mcts.get_result();

                let action = utils::sample(&policy, &mut self.rand);
                mcts.play(action)?;

                history.push((game, value, policy));
            }
        }

        self.state.0.store(MctsState::USABLE, Ordering::Relaxed);
        Ok(())
    }

    /// Retrieves the final results (score and policy) for any MCTS instances that have completed
    /// their search (game is finished at the root).
    ///
    /// Completed instances are removed from the batch. This method can only be called when
    /// the batch is in a `Usable` state.
    ///
    /// # Returns
    /// `Ok(Vec<(f64, [f64; N])>)` containing tuples of final score and action probabilities
    /// for completed games. Returns an empty `Vec` if no instances have finished.
    /// `Err(MctsError::InvalidState(_))` if the batch processor is not in the `Usable` state.
    pub fn next(&mut self) -> Result<Vec<History<T, N>>, MctsError>{
        match self.state.0.compare_exchange(MctsState::USABLE, MctsState::LOCKED, Ordering::SeqCst, Ordering::SeqCst){
            Ok(_) => {}
            Err(current_state) => { return Err(MctsError::InvalidState(current_state)); }
        }

        let mut result = Vec::new();

        for opt in &mut self.instances{
            if let Some((mcts, _history)) = &opt{
                if mcts.get_game().is_finish(){
                    {
                        let (mcts, history) = opt.as_mut().unwrap();

                        let mut score: f64 = -mcts.get_game().get_result().unwrap();
                        for (_game, value, _policy) in history.iter_mut().rev(){
                            *value=score;
                            score = -score;
                        }
                    }
                    result.push(opt.take().unwrap().1);
                    self.count -= 1;
                }
                else{
                    let (mcts, history) = opt.as_mut().unwrap();
                    
                    let (value, policy) = mcts.get_result();
                    history.push((mcts.get_game().clone(), value, policy));

                    let action: usize = utils::sample(&policy, &mut self.rand);

                    mcts.play(action)?;
                }
            }
        }

        self.state.0.store(MctsState::USABLE, Ordering::Relaxed);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use crate::{test_utils::{GameEvaluatorTest2, GameTest}, Game, MctsBatch, MctsError};

    #[test]
    fn test_batch_iterate_1() -> Result<(), MctsError>{
        let mut manager = MctsBatch::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        let a: GameTest = GameTest::new();
        let b: GameTest = GameTest::new();

        manager.populate_from_game(vec![a, b])?;

        let mut i: usize = 0;
        let mut result = Vec::new();
        while result.is_empty() {
            manager.iterate(&evaluator)?;
            result = manager.next()?;

            i += 1;
        }

        assert_eq!(result.len(), 2);
        assert_eq!(i, 4+1);

        Ok(())
    }

    #[test]
    fn test_batch_iterate_2() -> Result<(), MctsError>{
        let mut manager = MctsBatch::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        let a: GameTest = GameTest::new();
        let mut b: GameTest = GameTest::new();
        b.play(4);

        manager.populate_from_game(vec![a, b])?;
        assert_eq!(manager.get_count(), 2);

        let mut i: usize = 0;
        let mut result = Vec::new();
        while result.is_empty() {
            manager.iterate(&evaluator)?;
            result = manager.next()?;

            i += 1;
        }

        assert_eq!(result.len(), 1);
        assert_eq!(i, 3+1);
        assert_eq!(manager.get_count(), 1);

        manager.iterate(&evaluator)?;
        result = manager.next()?;

        assert_eq!(result.len(), 1);
        assert_eq!(manager.get_count(), 0);
        Ok(())
    }

    #[test]
    fn test_batch_iterate_3() -> Result<(), MctsError>{
        let mut manager = MctsBatch::<GameTest, 4>::new();
        let evaluator: GameEvaluatorTest2 = GameEvaluatorTest2::new();

        manager.populate_from_game(vec![GameTest::new()])?;

        let mut i: usize = 0;
        let mut result = Vec::new();
        while result.is_empty() {
            for _ in 0..18{
                manager.iterate(&evaluator)?;
            }
            result = manager.next()?;

            i += 1;
        }

        assert_eq!(result.len(), 1);
        assert_eq!(i, 4+1);

        assert_eq!((result[0][0].0).get_actions(), [true, true, true, true]);
        assert_eq!((result[0][0].1), 1.0);
        Ok(())
    }
}
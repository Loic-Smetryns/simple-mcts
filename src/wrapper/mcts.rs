use crate::{Action, Engine, MctsEngineError, ScoreTransformer, SelectionFunction, SelectionResult, StateEvaluation};

/// Configuration for the MCTS batch manager.
///
/// Contains the hyperparameters and functions required to run the selection
/// and backpropagation phases.
#[derive(Debug)]
pub struct MctsConfig<F: SelectionFunction, S: ScoreTransformer>{
    c: f32,
    selection_function: F,
    score_transformer: S,
}

impl<F: SelectionFunction, S: ScoreTransformer> MctsConfig<F, S> {
    /// Creates a new configuration for the MCTS engine.
    ///
    /// # Arguments
    ///
    /// * `c` - The exploration hyperparameter (often sqrt(2)).
    /// * `selection_function` - The heuristic used to select nodes (e.g., UCB1 or PUCT).
    /// * `score_transformer` - The function applied to scores during backpropagation.
    ///
    /// # Returns
    ///
    /// A new `MctsConfig` instance.
    pub fn new(c: f32, selection_function: F, score_transformer: S) -> Self {
        Self { c, selection_function, score_transformer }
    }
}

/// A unique, strongly-typed identifier representing a specific `Engine` instance
/// within the `MctsBatch` manager.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MctsId(usize);

/// Represents the output of a selection phase for a specific engine.
pub struct MctsSelectionResult {
    pub id: MctsId,
    pub selection: SelectionResult,
    pub path: Vec<Action>,
}

/// A structure bundling an MCTS evaluation with its targeted engine ID.
pub struct MctsStateEvaluation<const N: usize> {
    pub id: MctsId,
    pub selection: SelectionResult,
    pub evaluation: StateEvaluation<N>
}

/// A structure bundling the final score array with its targeted engine ID.
pub struct MctsScore<const N: usize> {
    pub id: MctsId,
    pub scores: [i32; N]
}

/// A structure bundling a game action with its targeted engine ID.
pub struct MctsAction {
    pub id: MctsId,
    pub action: Action
}

/// Errors that can occur when interacting with the `MctsBatch` manager.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum MctsError{
    /// An error occurred deep within the individual MCTS tree engine.
    EngineError(MctsEngineError, MctsId),
    /// The provided `MctsId` does not correspond to any active engine.
    InvalidMctsId(MctsId),
}

type EngineCollection<const N: usize> = Vec<Option<Engine<N>>>;
pub type IdCollection = Vec<MctsId>;
pub type ResultCollection = Vec<MctsSelectionResult>;
pub type StateEvaluationCollection<const N: usize> = Vec<MctsStateEvaluation<N>>;
pub type ScoreCollection<const N: usize> = Vec<MctsScore<N>>;
pub type ActionCollection = Vec<MctsAction>;

/// The Batch Manager for MCTS Engines.
///
/// It acts as a memory-efficient Arena Allocator (Slot Map) that can manage
/// thousands of independent MCTS trees simultaneously without reallocating memory.
pub struct MctsBatch<F: SelectionFunction, S: ScoreTransformer, const N: usize> {
    config: MctsConfig<F, S>,
    engines: EngineCollection<N>,
    free_list: Vec<MctsId>,

    sessions: usize
}

impl<F: SelectionFunction, S: ScoreTransformer, const N: usize> MctsBatch<F, S, N> {
    /// Initializes a new, empty batch manager from a given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The `MctsConfig` specifying exploration rate and heuristics.
    ///
    /// # Returns
    ///
    /// A new, empty `MctsBatch` instance.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::{MctsBatch, MctsConfig};
    /// let config = MctsConfig::new(1.414, |w, n, p_n, p, c| 0.0, |s| -s);
    /// let mut batch = MctsBatch::<_, _, 9>::from_config(config);
    /// ```
    pub fn from_config(config: MctsConfig<F, S>) -> Self {
        Self {
            config,
            engines: Vec::new(),
            sessions: 0,
            free_list: Vec::new(),
        }
    }

    /// Allocates a new MCTS engine in the batch.
    ///
    /// Memory slots are recycled using a free-list to guarantee O(1) amortized performance.
    ///
    /// # Returns
    ///
    /// Returns the uniquely generated `MctsId` for the new engine.
    pub fn add(&mut self) -> MctsId{
        self.sessions += 1;

        if let Some(id) = self.free_list.pop(){
            self.engines[id.0] = Some(Engine::new());
            id
        }
        else{
            let id = self.engines.len();
            self.engines.push(Some(Engine::new()));
            MctsId(id)
        }
    }

    /// Pre-allocates and populates the batch with a specified number of new engines.
    ///
    /// # Arguments
    ///
    /// * `n` - The number of engines to instantiate.
    ///
    /// # Returns
    ///
    /// A collection (`IdCollection`) containing all the newly minted `MctsId`s.
    pub fn populate(&mut self, n: usize) -> IdCollection {
        let mut ids = Vec::with_capacity(n);

        for _ in 0..n {
            ids.push(self.add());
        }

        ids
    }

    /// Safely deallocates a specific engine and marks its memory slot for reuse.
    ///
    /// Silently ignores the operation if the provided ID is already deleted or out of bounds.
    ///
    /// # Arguments
    ///
    /// * `id` - The `MctsId` of the engine to remove.
    pub fn remove_one(&mut self, id: MctsId){
        if let Some(slot) = self.engines.get_mut(id.0){
            if slot.take().is_some() {
                self.sessions -= 1;
                self.free_list.push(id);
            };
        }
    }

    /// Deallocates a collection of engines.
    ///
    /// # Arguments
    ///
    /// * `ids` - The collection of `MctsId`s to remove from the batch.
    pub fn remove(&mut self, ids: IdCollection) {
        for id in ids {
            self.remove_one(id);
        }
    }

    /// Deallocates all active engines and resets the free-list.
    ///
    /// # Note
    ///
    /// This completely invalidates any `MctsId` currently held by the user.
    pub fn clear(&mut self) {
        self.engines.clear();
        self.free_list.clear();
        self.sessions = 0;
    }

    /// Returns the number of currently active engines in the batch.
    ///
    /// # Returns
    ///
    /// An `usize` representing the active session count.
    pub fn sessions(&self) -> usize {
        self.sessions
    }

    /// Triggers the selection phase across all active engines.
    ///
    /// # Returns
    ///
    /// A `ResultCollection` containing the traversal paths and selection outcomes
    /// for every active engine.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::{MctsBatch, MctsConfig};
    /// # let config = MctsConfig::new(1.414, |w, n, p_n, p, c| 0.0, |s| -s);
    /// # let mut batch = MctsBatch::<_, _, 9>::from_config(config);
    /// batch.populate(3);
    /// let selections = batch.selection();
    /// assert_eq!(selections.len(), 3);
    /// ```
    pub fn selection(&self) -> ResultCollection {
        self.engines.iter().enumerate()
            .filter_map(|(id, opt)| {
                let engine = opt.as_ref()?;

                let mut path = Vec::new();
                let selection = engine.select(&mut path, &self.config.selection_function, self.config.c);

                Some(MctsSelectionResult {
                    id: MctsId(id),
                    selection,
                    path
                })
            }).collect()
    }

    /// Updates a single engine with an evaluation result using a custom, localized score transformer.
    ///
    /// This method acts as an "escape hatch" for complex or asymmetric games (e.g., 3-player games,
    /// hidden information) where the backpropagation logic depends on the specific context of the
    /// ongoing game, rather than the universal rule defined in `MctsConfig`.
    ///
    /// # Arguments
    ///
    /// * `evaluation` - The evaluated state to be integrated into the specific tree.
    /// * `score_transformer` - A mutable reference to a custom closure or struct implementing `ScoreTransformer`.
    ///
    /// # Errors
    ///
    /// Returns an `MctsError::InvalidMctsId` if the ID doesn't point to an active engine.
    /// Returns an `MctsError::EngineError` if the internal tree expansion or backpropagation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::{MctsBatch, MctsConfig, MctsStateEvaluation, StateEvaluation};
    /// # let config = MctsConfig::new(1.414, |w, n, p_n, p, c| 0.0, |s| -s);
    /// # let mut batch = MctsBatch::<_, _, 2>::from_config(config);
    /// # let id = batch.add();
    /// # let mut selections = batch.selection();
    /// # let sel = selections.pop().unwrap();
    /// let eval = MctsStateEvaluation {
    ///     id,
    ///     selection: sel.selection,
    ///     evaluation: StateEvaluation::Terminal(1.0)
    /// };
    ///
    /// // Using a custom transformer capturing a local state/context
    /// let mut local_scores = [1.0, -0.5, -0.5]; // e.g., 3-player score array
    ///
    /// batch.update_one_with_transformer(eval, &mut |score| {
    ///     // Custom asymmetric logic using the local context
    ///     score * local_scores[0]
    /// }).unwrap();
    /// ```
    pub fn update_one_with_transformer(&mut self, evaluation: MctsStateEvaluation<N>, score_transformer: &mut impl ScoreTransformer) -> Result<(), MctsError> {
        let opt = self.engines.get_mut(evaluation.id.0);

        if let Some(Some(engine)) = opt {
            engine
                .update(evaluation.evaluation, evaluation.selection, score_transformer)
                .map_err(|err| MctsError::EngineError(err, evaluation.id))?
        }
        else{
            return Err(MctsError::InvalidMctsId(evaluation.id));
        }

        Ok(())
    }

    /// Updates a single engine with an evaluation result and backpropagates the score.
    ///
    /// # Arguments
    ///
    /// * `evaluation` - The evaluated state to be integrated into the specific tree.
    ///
    /// # Errors
    ///
    /// Returns an `MctsError::InvalidMctsId` if the ID doesn't point to an active engine.
    /// Returns an `MctsError::EngineError` if the internal tree expansion or backpropagation fails.
    pub fn update_one(&mut self, evaluation: MctsStateEvaluation<N>) -> Result<(), MctsError> {
        let opt = self.engines.get_mut(evaluation.id.0);

        if let Some(Some(engine)) = opt {
            engine
                .update(evaluation.evaluation, evaluation.selection, &mut self.config.score_transformer)
                .map_err(|err| MctsError::EngineError(err, evaluation.id))?
        }
        else{
            return Err(MctsError::InvalidMctsId(evaluation.id));
        }

        Ok(())
    }

    /// Transactionally updates multiple engines with their respective evaluations.
    ///
    /// This function consumes the provided vector. If an error occurs, the function
    /// aborts and pushes the problematic evaluation back into the vector to prevent data loss.
    ///
    /// # Arguments
    ///
    /// * `evaluations` - A mutable reference to a collection of evaluations to process.
    ///
    /// # Errors
    ///
    /// Returns an `MctsError` if any single engine update fails or if an ID is invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # use crate::simple_mcts::{MctsBatch, MctsConfig, MctsStateEvaluation, StateEvaluation};
    /// # let config = MctsConfig::new(1.414, |w, n, p_n, p, c| 0.0, |s| -s);
    /// # let mut batch = MctsBatch::<_, _, 2>::from_config(config);
    /// let id = batch.add();
    /// let mut selections = batch.selection();
    /// let sel = selections.pop().unwrap();
    ///
    /// let mut evals = vec![MctsStateEvaluation {
    ///     id,
    ///     selection: sel.selection,
    ///     evaluation: StateEvaluation::Terminal(1.0)
    /// }];
    ///
    /// batch.update(&mut evals).unwrap();
    /// ```
    pub fn update(&mut self, evaluations: &mut StateEvaluationCollection<N>) -> Result<(), MctsError> {
        while let Some(evaluation) = evaluations.pop() {
            let opt = self.engines.get_mut(evaluation.id.0);

            if let Some(Some(engine)) = opt {
                engine
                    .update(evaluation.evaluation, evaluation.selection, &mut self.config.score_transformer)
                    .map_err(|err| {
                        let id = evaluation.id;
                        evaluations.push(evaluation);
                        MctsError::EngineError(err, id)
                    })?;
            }
            else{
                let id = evaluation.id;
                evaluations.push(evaluation);
                return Err(MctsError::InvalidMctsId(id));
            }
        }

        Ok(())
    }

    /// Retrieves the visit counts of all possible actions for a specific engine.
    ///
    /// # Arguments
    ///
    /// * `id` - The unique identifier of the target engine.
    ///
    /// # Returns
    ///
    /// Returns an array `[i32; N]` representing the visit distribution of the root node.
    ///
    /// # Errors
    ///
    /// Returns `MctsError::InvalidMctsId` if the specified engine is inactive or deleted.
    pub fn scores_one(&self, id: MctsId) -> Result<[i32; N], MctsError> {
        let opt = self.engines.get(id.0);

        if let Some(Some(engine)) = opt {
            Ok(engine.scores())
        }
        else{
            Err(MctsError::InvalidMctsId(id))
        }
    }

    /// Retrieves the visit counts of all possible actions for all active engines.
    ///
    /// # Returns
    ///
    /// A `ScoreCollection` containing the score arrays paired with their respective engine IDs.
    pub fn scores(&self) -> ScoreCollection<N> {
        self.engines.iter().enumerate()
            .filter_map(|(id, opt)| {
                let engine = opt.as_ref()?;
                Some(MctsScore {
                    id: MctsId(id),
                    scores: engine.scores()
                })
            }).collect()
    }

    /// Transactionally commits a list of played actions, updating the internal roots
    /// of the corresponding engines.
    ///
    /// # Arguments
    ///
    /// * `actions` - A mutable reference to the collection of actions to commit.
    ///
    /// # Errors
    ///
    /// Returns `MctsError::InvalidMctsId` if an action targets a non-existent engine.
    ///
    /// # Panics
    ///
    /// Panics if an action index is out of bounds (i.e., `>= N`) or if the internal
    /// tree structure of an engine is corrupted.
    pub fn commit_actions(&mut self, actions: &mut ActionCollection) -> Result<(), MctsError> {
        while let Some(action) = actions.pop() {
            if let Some(Some(engine)) = self.engines.get_mut(action.id.0) {
                engine.commit_action(action.action);
            }
            else {
                let id = action.id;
                actions.push(action);
                return Err(MctsError::InvalidMctsId(id));
            }
        }

        Ok(())
    }

    /// Commits a single played action, moving the root of the specified engine.
    ///
    /// # Arguments
    ///
    /// * `action` - The `MctsAction` containing the target ID and the action index.
    ///
    /// # Errors
    ///
    /// Returns `MctsError::InvalidMctsId` if the target engine does not exist.
    ///
    /// # Panics
    ///
    /// Panics if the action index is out of bounds (i.e., `>= N`) or if the internal
    /// tree structure is corrupted.
    pub fn commit_action(&mut self, action: MctsAction) -> Result<(), MctsError> {
        if let Some(Some(engine)) = self.engines.get_mut(action.id.0) {
            engine.commit_action(action.action);
            Ok(())
        }
        else { Err(MctsError::InvalidMctsId(action.id)) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_selection(_: f32, _: i32, _: i32, _: f32, _: f32) -> f32 { 0.0 }
    fn dummy_score_transformer(score: f32) -> f32 { -score }

    fn create_test_batch() -> MctsBatch<impl SelectionFunction, impl ScoreTransformer, 9> {
        let config = MctsConfig::new(1.0, dummy_selection, dummy_score_transformer);
        MctsBatch::from_config(config)
    }

    #[test]
    fn test_lifecycle_add_and_populate() {
        let mut batch = create_test_batch();

        let id1 = batch.add();
        assert_eq!(id1, MctsId(0));
        assert_eq!(batch.sessions(), 1);

        let ids = batch.populate(3);
        assert_eq!(ids, vec![MctsId(1), MctsId(2), MctsId(3)]);
        assert_eq!(batch.sessions(), 4);
    }

    #[test]
    fn test_freelist_reuse() {
        let mut batch = create_test_batch();
        batch.populate(3);

        batch.remove_one(MctsId(1));
        assert_eq!(batch.sessions(), 2);

        let new_id = batch.add();
        assert_eq!(new_id, MctsId(1));
        assert_eq!(batch.sessions(), 3);
    }

    #[test]
    fn test_remove_idempotence() {
        let mut batch = create_test_batch();
        batch.populate(2);

        batch.remove_one(MctsId(0));
        assert_eq!(batch.sessions(), 1);
        batch.remove_one(MctsId(0));
        assert_eq!(batch.sessions(), 1);
        assert_eq!(batch.free_list.len(), 1);
    }

    #[test]
    fn test_clear() {
        let mut batch = create_test_batch();
        batch.populate(5);
        batch.remove_one(MctsId(2));

        batch.clear();
        assert_eq!(batch.sessions(), 0);
        assert_eq!(batch.engines.len(), 0);
        assert_eq!(batch.free_list.len(), 0);
    }

    #[test]
    fn test_selection_success() {
        let mut batch = create_test_batch();
        batch.populate(2);

        let selections = batch.selection();
        assert_eq!(selections.len(), 2);
        assert_eq!(selections[0].id, MctsId(0));
        assert_eq!(selections[1].id, MctsId(1));
    }

    #[test]
    fn test_update_and_scores_success() {
        let mut batch = create_test_batch();
        let id = batch.add();

        let mut selections = batch.selection();
        let sel = selections.pop().unwrap();

        let eval = MctsStateEvaluation {
            id,
            selection: sel.selection,
            evaluation: StateEvaluation::Terminal(1.0),
        };

        let mut evaluations = vec![eval];
        let result = batch.update(&mut evaluations);

        assert!(result.is_ok());
        assert!(evaluations.is_empty(), "The evaluation should have been completed.");

        let scores = batch.scores_one(id);
        assert!(scores.is_ok());
    }

    #[test]
    fn test_update_one_invalid_id() {
        let mut batch = create_test_batch();

        let eval = MctsStateEvaluation {
            id: MctsId(99),
            selection: SelectionResult::Empty,
            evaluation: StateEvaluation::Terminal(1.0),
        };

        let result = batch.update_one(eval);
        assert!(matches!(result, Err(MctsError::InvalidMctsId(MctsId(99)))));
    }

    #[test]
    fn test_update_batch_transactional_rollback_on_invalid_id() {
        let mut batch = create_test_batch();
        let id_valid = batch.add();

        let eval_valid = MctsStateEvaluation {
            id: id_valid,
            selection: SelectionResult::Empty,
            evaluation: StateEvaluation::Terminal(1.0),
        };
        let eval_invalid = MctsStateEvaluation {
            id: MctsId(99),
            selection: SelectionResult::Empty,
            evaluation: StateEvaluation::Terminal(-1.0),
        };

        let mut evaluations = vec![eval_valid, eval_invalid];

        let result = batch.update(&mut evaluations);

        assert!(matches!(result, Err(MctsError::InvalidMctsId(MctsId(99)))));
        assert_eq!(evaluations.len(), 2, "The transaction failed; no data must be lost.");
        assert_eq!(evaluations.last().unwrap().id, MctsId(99));
    }

    #[test]
    fn test_commit_actions_transactional_rollback() {
        let mut batch = create_test_batch();
        let id_valid = batch.add();

        let action_valid = MctsAction { id: id_valid, action: Action::new(0) };
        let action_invalid = MctsAction { id: MctsId(99), action: Action::new(0) };

        let mut actions = vec![action_valid, action_invalid];

        let result = batch.commit_actions(&mut actions);

        assert!(matches!(result, Err(MctsError::InvalidMctsId(MctsId(99)))));
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_scores_invalid_id() {
        let batch = create_test_batch();
        let result = batch.scores_one(MctsId(42));
        assert!(matches!(result, Err(MctsError::InvalidMctsId(MctsId(42)))));
    }

    #[test]
    fn test_update_one_with_transformer_success() {
        let mut batch = create_test_batch();
        let id = batch.add();

        let mut selections = batch.selection();
        let sel = selections.pop().unwrap();

        let eval = MctsStateEvaluation {
            id,
            selection: sel.selection,
            evaluation: StateEvaluation::Terminal(1.0),
        };

        let mut calls_count = 0;
        let my_local_multiplier = 5.0;

        let mut custom_transformer = |score: f32| {
            calls_count += 1;
            score * my_local_multiplier
        };

        let result = batch.update_one_with_transformer(eval, &mut custom_transformer);

        assert!(result.is_ok(), "The update should have succeeded.");
        assert_eq!(calls_count, 1, "The custom transformer was not called the correct number of times.");
    }

    #[test]
    fn test_update_one_with_transformer_invalid_id() {
        let mut batch = create_test_batch();

        let eval = MctsStateEvaluation {
            id: MctsId(99),
            selection: SelectionResult::Empty,
            evaluation: StateEvaluation::Terminal(1.0),
        };

        let mut custom_transformer = |score: f32| score;

        let result = batch.update_one_with_transformer(eval, &mut custom_transformer);

        assert!(matches!(result, Err(MctsError::InvalidMctsId(MctsId(99)))));
    }
}
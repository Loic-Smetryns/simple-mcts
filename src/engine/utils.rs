use std::cmp::Ordering;

/// Calculates the Predictor + Upper Confidence bound applied to Trees (PUCT) score.
///
/// This function balances exploitation (choosing the action with the highest known average
/// score) and exploration (choosing actions with high prior probabilities or few visits).
///
/// It is specifically designed to be robust against division by zero for unvisited nodes.
///
/// The underlying formula is:
/// `Q + c * P * sqrt(N) / (1 + n)`
/// where `Q` is the average value (`score / node_visits`), or `0.0` if the node is unvisited.
///
/// # Arguments
///
/// * `score` - The cumulative score evaluated for this child node.
/// * `node_visits` - The number of times this specific child node has been visited (`n`).
/// * `parent_visits` - The total number of times the parent node has been visited (`N`).
/// * `policy` - The prior probability of selecting this node, typically provided by a neural network (`P`).
/// * `c` - The exploration hyperparameter. A higher value forces the search to explore unvisited nodes.
///
/// # Returns
///
/// Returns the computed PUCT priority score as an `f32`.
///
/// # Examples
///
/// ```
/// // Assuming the function is in your current scope
/// # use crate::simple_mcts::puct;
///
/// let score = 1.5;
/// let node_visits = 2;
/// let parent_visits = 8;
/// let policy = 0.25;
/// let c = std::f32::consts::SQRT_2;
///
/// let priority = puct(score, node_visits, parent_visits, policy, c);
///
/// // The expected value is approximately 1.0833 (13/12)
/// assert!((priority - (13.0 / 12.0)).abs() < 1e-5);
/// ```
#[inline]
pub fn puct(score: f32, node_visits: i32, parent_visits: i32, policy: f32, c: f32) -> f32{
    let q = if node_visits > 0 { score / node_visits as f32 } else { 0.0 };
    q + c * policy * (parent_visits as f32).sqrt() / (1. + node_visits as f32)
}

/// Inverts the evaluation score during the backpropagation phase.
///
/// In zero-sum, alternating-turn games, a winning
/// position for the current player is a losing position for the opponent.
/// This function is used to negate the score as it propagates up the Monte Carlo
/// Tree Search (MCTS) path, ensuring that each node evaluates the game state
/// from the correct perspective of the player whose turn it is to act.
///
/// # Arguments
///
/// * `s` - The score evaluated from the child node's perspective.
///
/// # Returns
///
/// Returns the negated score (`-s`) as an `f32`.
///
/// # Examples
///
/// ```
/// // Assuming the function is in your current scope
/// # use crate::simple_mcts::negate_score;
///
/// // A guaranteed win for the child (+1.0) means a guaranteed loss for the parent (-1.0).
/// let child_score = 1.0;
/// let parent_score = negate_score(child_score);
///
/// assert_eq!(parent_score, -1.0);
/// ```
#[inline]
pub fn negate_score(s: f32) -> f32{
    -s
}

/// Converts a raw array of visit counts into a normalized probability distribution.
///
/// This function relies on a highly optimized, branchless implementation using a
/// microscopic epsilon (`1e-18`). This guarantees numerical stability and elegantly
/// prevents division by zero `NaN` errors. If a node has no visits, the function
/// naturally falls back to returning a perfectly uniform probability distribution.
///
/// # Arguments
///
/// * `visits` - An array of `i32` representing the accumulated MCTS visit counts for each action.
///
/// # Returns
///
/// An array of `f32` where each element represents the probability of choosing the
/// corresponding action. The sum of all elements is guaranteed to be `1.0` (within
/// standard `f32` floating-point precision).
///
/// # Examples
///
/// ```
/// // Assuming the function is in your current scope
/// # use crate::simple_mcts::visits_to_probabilities;
///
/// let visits = [0, 3, 1];
/// let probas = visits_to_probabilities(visits);
///
/// // The total visits is 4.
/// assert!(probas[0].abs() < 1e-6); // 0
/// assert!((probas[1] - 0.75).abs() < 1e-6); // 3/4
/// assert!((probas[2] - 0.25).abs() < 1e-6); // 1/4
/// ```
#[inline]
pub fn visits_to_probabilities<const N: usize>(visits: [i32; N]) -> [f32; N]{
    // constant under f32 precision => 1. + Epsilon = 1. but Epsilon != 0
    const EPSILON: f32 = 1e-18;

    // while N is lower than 10 billions, EPSILON * N is lower than f32 precision
    let sum = visits.iter().sum::<i32>() as f32 + (EPSILON * N  as f32);

    std::array::from_fn(|i| {
        (visits[i] as f32 + EPSILON) / sum
    })
}

/// Converts visit counts into a probability distribution using a temperature parameter.
///
/// This function follows the AlphaZero policy extraction formula where the probability
/// of an action is proportional to `N^(1/tau)`. The temperature parameter (`tau`)
/// controls the exploration/exploitation trade-off during the final move selection:
///
/// * `tau = 1.0`: Standard proportional selection (equivalent to `visits_to_probabilities`).
/// * `tau > 1.0`: Softens the distribution, pulling probabilities closer together to encourage exploration.
/// * `tau <= 0.0`: Pure greedy selection. Returns `1.0` for the action with the absolute maximum visits.
///
/// **Note on Greedy Selection:** If `temperature` is 0.0 or lower and multiple actions
/// share the exact same maximum visit count (a tie), the function safely distributes
/// the probability equally among the tied actions.
///
/// # Arguments
///
/// * `visits` - An array of `i32` representing the visit counts.
/// * `temperature` - The `tau` parameter controlling the sharpness of the distribution.
///
/// # Returns
///
/// An array of `f32` representing the temperature-scaled probability distribution.
///
/// # Examples
///
/// ```
/// // Assuming the function is in your current scope
/// # use crate::simple_mcts::visits_to_probabilities_with_temperature;
///
/// // Example of a greedy selection with a tie
/// let visits = [10, 50, 50, 5];
/// let probas = visits_to_probabilities_with_temperature(visits, 0.0);
///
/// // The probability is perfectly split between the two top choices
/// assert_eq!(probas, [0.0, 0.5, 0.5, 0.0]);
/// ```
#[inline]
pub fn visits_to_probabilities_with_temperature<const N: usize>(visits: [i32; N], temperature: f32) -> [f32; N]{
    // constant under f32 precision => 1. + Epsilon = 1. but Epsilon != 0
    const EPSILON: f32 = 1e-18;

    if temperature > 0.{
        let exponent = 1. / temperature;

        let mut sum = 0.;
        let mut visits = std::array::from_fn(|i| {
            let v = (visits[i] as f32).powf(exponent) + EPSILON;
            sum += v;
            v
        });

        visits.iter_mut().for_each(|v| *v /= sum);
        visits
    }
    else{
        let (max, count) = visits.iter().fold((0i32, 0usize), |(max, count), v|{
            match (*v).cmp(&max) {
                Ordering::Greater => (*v, 1),
                Ordering::Equal => (max, count+1),
                Ordering::Less => (max, count)
            }
        });

        let value = 1. / (count as f32);

        std::array::from_fn(|i| {
            if visits[i] == max { value } else { 0. }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! assert_approx_eq {
        ($a:expr, $b:expr) => {
            assert!(($a - $b).abs() < 1e-5, "left: {}, right: {}", $a, $b);
        };
    }

    #[test]
    fn test_puct_c_0() {
        let score = puct(1.5, 2, 8, 0.25, 0.);
        assert_approx_eq!(score, 0.75);
    }


    #[test]
    fn test_puct_policy_0() {
        let score = puct(1.5, 2, 8, 0., std::f32::consts::SQRT_2);
        assert_approx_eq!(score, 0.75);
    }

    #[test]
    fn test_puct_score_0() {
        let score = puct(0.0, 2, 8, 0.25, std::f32::consts::SQRT_2);
        assert_approx_eq!(score, 1./3.);
    }

    #[test]
    fn test_puct_visits_0() {
        let score = puct(0.0, 0, 8, 0.25, std::f32::consts::SQRT_2);
        assert_approx_eq!(score, 1.);
    }

    #[test]
    fn test_puct_1() {
        let score = puct(1.5, 2, 8, 0.25, std::f32::consts::SQRT_2);
        assert_approx_eq!(score, 13./12.);
    }

    #[test]
    fn test_puct_2() {
        let score = puct(-0.5, 5, 18, 0.75, std::f32::consts::SQRT_2);
        assert_approx_eq!(score, 0.65);
    }

    #[test]
    fn test_negate_score_positive_to_negative() {
        assert_eq!(negate_score(1.0), -1.0);
        assert_eq!(negate_score(42.5), -42.5);
    }

    #[test]
    fn test_negate_score_negative_to_positive() {
        assert_eq!(negate_score(-1.0), 1.0);
        assert_eq!(negate_score(-0.75), 0.75);
    }

    #[test]
    fn test_negate_score_zero() {
        assert_eq!(negate_score(0.0), 0.0);
        assert_eq!(negate_score(-0.0), 0.0);
    }

    fn assert_array_approx_eq<const N: usize>(actual: [f32; N], expected: [f32; N], threshold: f32) {
        for i in 0..N {
            assert!(
                (actual[i] - expected[i]).abs() < threshold,
                "Error at index {}: received value {}, expected value {}",
                i, actual[i], expected[i]
            );
        }
    }

    #[test]
    fn test_probabilities_normal_case() {
        let visits = [10, 30, 0, 10];
        let probas = visits_to_probabilities(visits);

        assert_array_approx_eq(probas, [0.2, 0.6, 0.0, 0.2], 1e-6);
    }

    #[test]
    fn test_probabilities_all_zeros() {
        let visits = [0, 0, 0, 0, 0];
        let probas = visits_to_probabilities(visits);

        assert_array_approx_eq(probas, [0.2, 0.2, 0.2, 0.2, 0.2], 1e-6);
    }

    #[test]
    fn test_temperature_normal_tau_1() {
        let visits = [10, 30];
        let probas = visits_to_probabilities_with_temperature(visits, 1.0);
        assert_array_approx_eq(probas, [0.25, 0.75], 1e-6);
    }

    #[test]
    fn test_temperature_high_tau_exploration() {
        let visits = [10, 90];
        let probas = visits_to_probabilities_with_temperature(visits, 10.0);
        assert_array_approx_eq(probas, [0.44528, 0.55471], 1e-4);
    }

    #[test]
    fn test_temperature_greedy_single_max() {
        let visits = [5, 100, 10, 0];
        let probas = visits_to_probabilities_with_temperature(visits, 0.0);
        assert_array_approx_eq(probas, [0.0, 1.0, 0.0, 0.0], 1e-6);
    }

    #[test]
    fn test_temperature_greedy_ties() {
        let visits = [10, 50, 50, 5];
        let probas = visits_to_probabilities_with_temperature(visits, 0.0);
        assert_array_approx_eq(probas, [0.0, 0.5, 0.5, 0.0], 1e-6);
    }

    #[test]
    fn test_temperature_greedy_triple_ties() {
        let visits = [100, 100, 100];
        let probas = visits_to_probabilities_with_temperature(visits, 0.0);
        assert_array_approx_eq(probas, [0.333333, 0.333333, 0.333333], 1e-5);
    }

    #[test]
    fn test_temperature_all_zeros_fallback() {
        let visits = [0, 0, 0, 0];
        let probas_greedy = visits_to_probabilities_with_temperature(visits, 0.0);
        let probas_soft = visits_to_probabilities_with_temperature(visits, 0.5);

        assert_array_approx_eq(probas_greedy, [0.25, 0.25, 0.25, 0.25], 1e-6);
        assert_array_approx_eq(probas_soft, [0.25, 0.25, 0.25, 0.25], 1e-6);
    }
}
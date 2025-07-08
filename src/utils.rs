use rand::{self, rngs::StdRng, Rng};

/// Samples an action index from a given policy distribution using a random number generator.
///
/// This function performs a weighted random selection, where actions with higher
/// policy probabilities are more likely to be chosen.
///
/// # Parameters
/// - `policy`: A slice representing the probability distribution over actions.
///             The sum of probabilities should ideally be 1.0.
/// - `rng`: A mutable reference to a `StdRng` (standard random number generator)
///          instance. This allows for reproducible sampling if the RNG is seeded.
///
/// # Returns
/// The index of the sampled action.
///
/// # Panics
/// Panics if the `policy` slice is empty and if no action is selected (should not happen
/// if policy sums to 1.0).
pub fn sample(policy: &[f64], rng: &mut StdRng) -> usize{
    let mut random: f64 = rng.random();

    policy.iter().position(|&x|{
        random -= x;
        random <= 0.
    }).unwrap_or(policy.len()-1)
}
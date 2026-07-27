use sha2::{Digest, Sha256};

use crate::{Hash32, TaskPolicy};

const SAMPLE_DOMAIN: &[u8] = b"pocomp/task-sampling/v1";

#[must_use]
pub fn is_sampled(seed: Hash32, epoch_id: &str, task_id: &str, policy: &TaskPolicy) -> bool {
    if policy.sample_denominator == 0 || policy.sample_numerator > policy.sample_denominator {
        return false;
    }
    if policy.sample_numerator == policy.sample_denominator {
        return true;
    }
    if policy.sample_numerator == 0 {
        return false;
    }

    let mut hasher = Sha256::new();
    hasher.update(SAMPLE_DOMAIN);
    hasher.update(seed.0);
    hasher.update((epoch_id.len() as u64).to_be_bytes());
    hasher.update(epoch_id.as_bytes());
    hasher.update((task_id.len() as u64).to_be_bytes());
    hasher.update(task_id.as_bytes());
    let digest: [u8; 32] = hasher.finalize().into();
    let draw = u64::from_be_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ]);
    let threshold = (u128::from(u64::MAX) + 1) * u128::from(policy.sample_numerator)
        / u128::from(policy.sample_denominator);
    u128::from(draw) < threshold
}

#[must_use]
pub fn sampled_task_ids<'a>(
    seed: Hash32,
    epoch_id: &str,
    task_ids: impl IntoIterator<Item = &'a str>,
    policy: &TaskPolicy,
) -> Vec<String> {
    task_ids
        .into_iter()
        .filter(|task_id| is_sampled(seed, epoch_id, task_id, policy))
        .map(str::to_owned)
        .collect()
}

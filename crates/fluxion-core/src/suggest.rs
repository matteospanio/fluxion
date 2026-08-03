//! "Did you mean …?" — the one helper behind every name error in the library.
//!
//! Kept here rather than in the CLI because the same suggestion has to reach every interface: a
//! typo in a chain string is the same mistake whether it arrives from a terminal, a Python call, a
//! C host or the browser. No dependency: `fluxion-core` builds offline on `serde` alone.

/// The closest candidate to `needle`, or `None` when nothing is close enough to be a likely typo.
///
/// The threshold is `max(1, longest / 3)` edits — generous enough for a plausible slip, tight
/// enough that an unrelated word gets no suggestion at all (a wrong suggestion is worse than none).
/// Ties go to the first candidate, so pass them in catalog order for a stable message.
///
/// ```
/// use fluxion_core::suggest::closest;
/// assert_eq!(closest("hipass", ["gain", "highpass", "lowpass"]), Some("highpass"));
/// assert_eq!(closest("nosuchthing", ["gain", "highpass"]), None);
/// ```
pub fn closest<'a, I>(needle: &str, candidates: I) -> Option<&'a str>
where
    I: IntoIterator<Item = &'a str>,
{
    if needle.is_empty() {
        return None;
    }
    let mut best: Option<(usize, &str)> = None;
    for candidate in candidates {
        let budget = (needle.chars().count().max(candidate.chars().count()) / 3).max(1);
        let distance = edit_distance(needle, candidate);
        if distance <= budget && best.is_none_or(|(d, _)| distance < d) {
            best = Some((distance, candidate));
        }
    }
    best.map(|(_, candidate)| candidate)
}

/// Edit distance counting an insertion, a deletion, a substitution **or a transposition** as one
/// edit (optimal string alignment). Transpositions matter: `gian` for `gain` is the single most
/// common typo, and plain Levenshtein scores it 2, far enough to lose the suggestion.
///
/// Three rolling rows, one allocation each.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    // Row 0: the cost of deleting the first `j` characters of `b`.
    let mut before = vec![0usize; b.len() + 1];
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];

    for i in 0..a.len() {
        curr[0] = i + 1;
        for j in 0..b.len() {
            let substitute = prev[j] + usize::from(a[i] != b[j]);
            let delete = prev[j + 1] + 1;
            let insert = curr[j] + 1;
            let mut best = substitute.min(delete).min(insert);
            if i > 0 && j > 0 && a[i] == b[j - 1] && a[i - 1] == b[j] {
                best = best.min(before[j - 1] + 1);
            }
            curr[j + 1] = best;
        }
        // Rotate: before <- prev <- curr.
        std::mem::swap(&mut before, &mut prev);
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::{closest, edit_distance};

    #[test]
    fn distance_matches_the_textbook_cases() {
        assert_eq!(edit_distance("", ""), 0);
        assert_eq!(edit_distance("", "abc"), 3);
        assert_eq!(edit_distance("abc", ""), 3);
        assert_eq!(edit_distance("abc", "abc"), 0);
        assert_eq!(edit_distance("kitten", "sitting"), 3);
        assert_eq!(edit_distance("hipass", "highpass"), 2);
        assert_eq!(edit_distance("cutof", "cutoff"), 1);
        // A transposition is one edit, not two.
        assert_eq!(edit_distance("gian", "gain"), 1);
        assert_eq!(edit_distance("comapnd", "compand"), 1);
    }

    #[test]
    fn suggests_a_plausible_typo() {
        let ops = ["gain", "lowpass", "highpass", "cutoff", "compand"];
        assert_eq!(closest("hipass", ops), Some("highpass"));
        assert_eq!(closest("lowpas", ops), Some("lowpass"));
        assert_eq!(closest("cutof", ops), Some("cutoff"));
        assert_eq!(closest("gian", ops), Some("gain"));
    }

    #[test]
    fn stays_quiet_when_nothing_is_close() {
        let ops = ["gain", "lowpass", "highpass"];
        // A wrong suggestion is worse than none, so an unrelated word gets nothing.
        assert_eq!(closest("nosuchthing", ops), None);
        assert_eq!(closest("reverb", ops), None);
        assert_eq!(closest("", ops), None);
        assert_eq!(closest("gain", std::iter::empty()), None);
    }

    #[test]
    fn picks_the_nearest_when_several_are_close() {
        // `lowshelf` is 1 edit away, `highshelf` is 4 — the nearest wins.
        assert_eq!(
            closest("lowshelv", ["highshelf", "lowshelf"]),
            Some("lowshelf")
        );
    }
}

//! Minimal CSV row splitting, shared by the ASL and trajectory readers.
//!
//! Deliberately not a CSV *parser*. The dataset formats this crate reads —
//! EuRoC, TUM VI, TUM trajectories — are numeric tables with no quoting, no
//! embedded separators and no escapes, so splitting on the separator is both
//! sufficient and faster than a general reader. Pulling in a full CSV crate
//! would buy nothing these formats can use.

/// Split `content` into rows of trimmed fields, skipping blank and `#` lines.
///
/// The returned row index is the position among *non-comment* rows, not the
/// physical file line. That matches how the ASL reader has always numbered its
/// errors, and it is the number a user counting data rows expects.
///
/// # Arguments
///
/// * `content` — the whole file.
/// * `separator` — `,` for ASL CSV, whitespace-splitting when `None` (TUM).
pub(crate) fn split_rows(content: &str, separator: Option<char>) -> Vec<Vec<String>> {
    content
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty() && !trimmed.starts_with('#')
        })
        .map(|line| match separator {
            Some(sep) => line
                .split(sep)
                .map(|field| field.trim().to_owned())
                .collect(),
            None => line
                .split_whitespace()
                .map(|field| field.to_owned())
                .collect(),
        })
        .collect()
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;

    #[test]
    fn blank_and_comment_lines_are_skipped() {
        let rows = split_rows("# header\n\n1,2\n\n  \n3,4\n", Some(','));
        assert_eq!(rows, vec![vec!["1", "2"], vec!["3", "4"]]);
    }

    #[test]
    fn fields_are_trimmed_when_splitting_on_a_separator() {
        let rows = split_rows(" 1 , 2 ,3 \n", Some(','));
        assert_eq!(rows, vec![vec!["1", "2", "3"]]);
    }

    /// TUM files are whitespace-separated with runs of spaces, so consecutive
    /// separators must not produce empty fields the way `split(' ')` would.
    #[test]
    fn whitespace_splitting_collapses_runs() {
        let rows = split_rows("1   2\t3\n", None);
        assert_eq!(rows, vec![vec!["1", "2", "3"]]);
    }

    /// A comment marker only counts at the start of a line; `#` inside a field
    /// is data.
    #[test]
    fn a_hash_after_the_first_column_is_not_a_comment() {
        let rows = split_rows("1,2#3\n", Some(','));
        assert_eq!(rows, vec![vec!["1", "2#3"]]);
    }
}

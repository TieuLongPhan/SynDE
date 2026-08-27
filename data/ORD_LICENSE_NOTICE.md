# Open Reaction Database attribution

The following tracked data files are adapted from the Open Reaction Database
`ord-data` repository:

- `ord_training_candidates.csv`
- `ord_external_candidates.csv`
- `ord_cohort_selection.audit.json`
- `ord_training_xtb.csv`
- `ord_test_xtb.csv`
- `ord_xtb_results.sqlite3`

Source: <https://github.com/open-reaction-database/ord-data>

Pinned source commit: `ff7427ff6e65e9c5cf25caeccb9ed6407179d3d6`

Upstream data license: Creative Commons Attribution-ShareAlike 4.0
International (CC-BY-SA-4.0), <https://creativecommons.org/licenses/by-sa/4.0/>.

Changes made by the SynDE project include extraction of molecular identifiers,
separation of disconnected components, RDKit 2025.09.6 canonicalization,
filtering to a declared neutral closed-shell element and size domain,
deduplication, molecular-formula grouping, deterministic diversity sampling,
and label-blind formula-disjoint partitioning. Exact processing rules, source
hashes, and output hashes are recorded in `ord_cohort_selection.audit.json` and
the scripts under `Experiment/scripts/`.

These adapted data files are distributed under CC-BY-SA-4.0. No endorsement by
the Open Reaction Database contributors is implied.

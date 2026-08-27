# Pinned ORD molecular inventory

The active data workflow starts from the official Open Reaction Database
repository at the immutable commit recorded in
`Experiment/scripts/01_prepare_ord.py`.

## Repository layout

The label-blind cohort definitions and the recovered compact xTB results are
versioned. Large upstream source and preparation scratch files remain local:

| Status | Files | Purpose |
|---|---|---|
| Tracked | `ord_training_candidates.csv`, `ord_external_candidates.csv` | frozen calculation manifests |
| Tracked | `ord_cohort_selection.audit.json` | selection rules, hashes, and split firewall |
| Tracked | `ord_training_xtb.csv`, `ord_test_xtb.csv` | recovered xTB result snapshots, including persistent failures |
| Tracked | `ord_xtb_results.sqlite3` | complete resumable xTB result database |
| Generated | `ord-data/`, `ord_raw*`, `ord.csv` | upstream checkout and resumable preparation scratch |

The published benchmark uses exactly the training and external-validation
cohorts; there is no third evaluation dataset.

Run from the repository root:

```bash
conda env update -n synde -f env.yml
conda run -n synde python Experiment/scripts/01_prepare_ord.py all
```

The stages may also be run separately with `fetch`, `extract`, and
`canonicalize`. Extraction checkpoints each source dataset in
`data/.ord_raw_work_v3.sqlite3`; if it is interrupted, rerun `extract` and it will
resume from the last committed dataset. Each protobuf is decoded in a separate
process so its memory is released before the next file.

## Generated files

- `data/ord-data/`: ignored local checkout of the pinned official ORD commit.
- `data/ord_raw.csv`: unique raw molecular strings extracted from structured
  reaction inputs/products and reaction-SMILES identifiers, with representative
  ORD provenance.
- `data/ord.csv`: RDKit-canonicalized molecular components. Disconnected
  records are split with `Chem.GetMolFrags`; atom-map numbers are removed.
- `data/ord_formula_groups.csv`: counts by molecular formula and formal charge.
- `data/ord_raw.audit.json` and `data/ord.audit.json`: source commit, hashes,
  RDKit version, rules, and complete processing counts.
- `data/ord_training_candidates.csv` and
  `data/ord_external_candidates.csv`: tracked, label-blind frozen manifests for
  the 98,479-molecule calculation.
- `data/ord_cohort_selection.audit.json`: tracked selection rules, counts,
  leakage checks, and manifest hashes.

`ord.csv` deliberately retains parseable charged, radical, and unsupported
components. The `basic_eligible`, `paper_eligible`, and `exclusion_reason`
columns distinguish the neutral closed-shell element domain used by the SynDE
experiment without silently deleting the rest of ORD.

The cohort includes H, B, C, N, O, F, Si, P, S, Cl, Br, and I. Hydrogen is
normally implicit in input SMILES but is included in both the formula and the
recorded element set. Ordinary explicit hydrogen is normalized to its implicit
representation before identity and feature generation. Any atom with a
nonzero isotope number, including `[2H]` and `[3H]`, is excluded under the
`isotopically_labelled` reason. Elemental calibration and validation must use
this active cohort.

This preparation step does not run xTB. Cohort selection and reference-energy
generation must consume only `paper_eligible=1` rows and preserve complete
formula groups.

## Label-blind cohort manifests

After inspecting `ord.audit.json` and choosing an xTB budget, create immutable
candidate manifests with:

```bash
conda run -n synde python Experiment/scripts/02_select_cohorts.py \
  --maximum-groups 15000 \
  --maximum-per-group 10 \
  --maximum-heavy-atoms 30 \
  --validation-fraction 0 \
  --external-fraction 0.20
```

Omit `--maximum-groups` to assign every rankable formula group. The active
command sets the development-validation fraction to zero and uses an 80%
training / 20% external-validation assignment by a fixed SHA-256 formula-group
split.
Formula groups and canonical connectivities cannot cross partitions. Up to ten
molecules per formula are retained by a deterministic ECFP4 max-min diversity
rule. Selection uses no energy values and must be frozen before any xTB
calculation begins.

The amended frozen cohort contains 98,479 candidate calculations: 78,534 training
molecules in 11,995 formula groups and 19,945 external molecules in 3,005
groups. Hyperparameters are chosen by formula-grouped cross-validation within
training, so a separately labelled validation partition is unnecessary.

Because all three partitions come from one ORD snapshot, the 20% partition is
a same-source external validation, not a fully independent source. Use a later
ORD increment or another molecular collection for a stronger external test;
apply the same eligibility rules and exclude every training formula and
connectivity before labels are inspected. Report scaffold-unseen and
fingerprint-distance subsets as transfer diagnostics rather than silently
redefining the primary cohort.

With `--scaffold-disjoint`, formula groups that share any nonempty achiral
Bemis--Murcko scaffold are joined into a connected component and that whole
component is assigned to one partition. Empty scaffolds are ignored because
otherwise every acyclic molecule would collapse into a single component; use
the reported fingerprint-distance diagnostics for those molecules.

Exact scaffold-component splitting is not used for the primary 98k cohort:
transitive scaffold links create a dominant component and a strong molecular-
size shift. Instead, evaluate the formula-disjoint external cohort as primary and
report the prespecified scaffold-unseen subset as a transfer diagnostic.

## Running xTB with at most 16 CPUs

The candidate CSVs and their selection audit are tracked in the repository.
The calculation database and energy CSVs are tracked recovery artifacts. Check
that every manifest record is present without launching xTB:

```bash
conda run -n synde python Experiment/scripts/03_run_xtb.py \
  --workers 16 --dry-run
```

Run a resumable 600-molecule timing/failure pilot:

```bash
conda run -n synde python Experiment/scripts/03_run_xtb.py \
  --workers 16 --limit 600
```

Then resume all remaining candidates by omitting `--limit`:

```bash
conda run -n synde python Experiment/scripts/03_run_xtb.py --workers 16
```

On the committed snapshot the dry run reports zero pending candidates. The
database contains all 98,479 candidates: 78,517 successful training labels,
19,940 successful external labels, and the 17 training plus 5 external
failures retained for provenance. Four successful training calculations are
excluded later because their two formula groups are no longer rankable, giving
the published 78,513/19,940 analysis counts.

xTB is CPU-based here. The runner enforces a maximum of 16 process workers and
sets xTB plus common numerical backends to one thread per worker, preventing
OpenMP oversubscription. SQLite commits each completed molecule, so rerunning
the command resumes safely. Only compact identity, status, energy, runtime, and
protocol metadata are retained; full logs and XYZ files are discarded.

## Data attribution and license

The candidate manifests are adapted from the Open Reaction Database `ord-data`
repository at commit `ff7427ff6e65e9c5cf25caeccb9ed6407179d3d6` and are
distributed under CC-BY-SA-4.0. See `data/ORD_LICENSE_NOTICE.md`. The repository's
MIT license continues to apply to SynDE software, not to these adapted ORD data
files.

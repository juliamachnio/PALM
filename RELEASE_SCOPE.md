# PALM public-release scope

This repository is prepared in small, reviewable local commits before any
change is pushed to the public `juliamachnio/PALM` repository. The release
branch is intentionally unpushed until each commit is accepted.

## Planned public additions

- **ALDA**: PALM-based learning-curve fitting, export, and deployment-advisor
  analysis, with reproducible configurations and documentation.
- **Mechanism-driven theory**: computational proxies, phase/regime analysis,
  and the proxy-derived `hard_switch` active-learning baseline.
- Reproducibility support: public dataset acquisition instructions,
  deterministic split generation and validation, and versioned split
  manifests. The datasets themselves are never included.

## Explicit exclusions

Do not add any of the following to this repository or its Git history:

- Raw, processed, or patient-derived datasets; private dataset identifiers;
  credentials; access tokens; or institution-specific filesystem paths.
- Training checkpoints, experiment outputs, raw logs, Slurm output, large
  result arrays, or unpublished figures/tables.
- ALPS, urgency estimation, soft-transition/soft-allocation methods,
  rebuttal code, or other work not required by the ALDA or mechanism-theory
  papers.

## Review checklist for every later commit

1. State which paper component the commit supports.
2. Confirm that every new path is portable and has no local absolute path.
3. Confirm that no excluded file or unrelated research module is included.
4. Run the relevant focused validation and report its result before review.

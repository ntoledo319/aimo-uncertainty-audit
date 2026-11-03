# Uncertainty profiling with a constant fallback

Reproduction material for Codabench AIMO Interpretability Challenge Main Track **928050** and Small Models Track **929716**. No new score or improved model is claimed.

This repository contains participant changes, an exact provenance manifest, and retained experiment evidence. It **does not redistribute the upstream implementation, trained artifact, original submission ZIPs, language-model weights, datasets, environment or credentials**.

## Prior work and attribution

The method is the organizers' generation-uncertainty baseline, including its fitted RandomForest and constant non-robust fallback. Pin:

https://github.com/aimo-interp/getting-started/tree/995f9923a680a157c9239e08e4ef0fa0fa53bb58

Six of eight submitted files are exactly identical to that source. The only participant changes are shown in `submission.patch`: an explicit (behaviorally redundant) fallback check in `solution.py` and a CPU-only dtype change from float32 to bfloat16 in `uncertainty_profile/extraction.py`. `provenance.json` records verified upstream and submitted SHA-256 hashes for every file. The local upstream copies were also checked against the pinned GitHub tree's Git blob IDs.

The separate text-only training script and its saved candidate scores document an alternative that was rejected, not code present in the competition entry. Cross-validation scores selected both threshold and model family on the same out-of-fold data; they are not unbiased independent estimates. The report explains this and the evidence limitations.

The patch preserves comments from the frozen submission for byte-exact reconstruction. Its phrase “accuracy-optimal under label skew” is not a guarantee on unseen data; constant-fallback accuracy equals the prevalence of non-robust labels. Its CPU/GPU dtype equivalence comment is a historical assertion, not a verified cross-hardware claim. The report explicitly limits these conclusions and does not promise bitwise hardware independence.

## Reconstruct without training or inference

Requires Python's standard library and the `patch` command. Prefer an existing checkout:

```bash
python reconstruct.py --upstream /path/to/getting-started --output /new/path/entry
```

Or explicitly fetch only the eight needed files from the pinned upstream commit (approximately 1 MB artifact plus source):

```bash
python reconstruct.py --download --output /new/path/entry
```

The script checks upstream hashes, applies the tiny patch, and checks every resulting byte against the frozen submitted contents. It creates main and small-model ZIPs with identical file contents to the entry (ZIP timestamps/metadata can differ). It refuses to overwrite an existing output directory. It never installs dependencies, runs a model, or submits a forecast/competition entry.

For evaluation, use the upstream fixed competition runtime and harness. Particularly, the artifact was serialized with scikit-learn 1.8.0. The full 24-case development split has 15 non-robust and nine robust cases. The 23-case fallback-only subset omits `qwen3-8b:low`, leaving 14 non-robust and nine robust cases. Stored scoring files confirm 14/23 accuracy and 1.0 output coverage for both tracks on this subset; they do not demonstrate the learned artifact's benefit. See the accompanying report for exact dataset revisions, inference settings and reproduction instructions. No completed full-length local evaluation is claimed.

## Rights and AI disclosure

Upstream authors retain rights to their implementation and fitted artifact. The inspected source snapshot has no license file and GitHub detected no SPDX license on November 2, 2026. This repository does not make a blanket license grant over upstream work; consult its maintainers and competition terms for reuse permissions. The patch records the participant's small changes without claiming authorship of the underlying method.

Autonomous AI agents assembled the entry and wrote the report. A Codex agent prepared this reproducibility package from preserved artifacts on November 2, 2026. No new training, generation or private-test experiments were run, and no human technical review is asserted. Author identity and competition submission are handled separately by the human participant's account.

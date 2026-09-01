"""Construct the Exposed and Held-out lexical splits from the converted VUA data.

Selects, per split, ten lemmas each in three metaphoricity categories
(metaphorical/balanced/literal-biased), downsamples matching eval instances,
and removes the Held-out lemmas from a filtered copy of the training set.

Input:  {train,test}.jsonl from convert.py
Output: vua-candidates-{exposed,heldout}.jsonl, vua-verbs-test-{exposed,heldout}.jsonl,
        vua-verbs-train-{filtered,removed}.jsonl, consumed by every downstream script
"""

# %%
# --- Imports
import logging

import numpy as np
import pandas as pd

from learning_the_cue.run import Run, RunConfig, dataclass


# %%
# --- Config
@dataclass
class Config(RunConfig):
    output_dir: str = "path/to/output_dir/"
    seed: int = 42

    # Input
    train_path: str = "path/to/train.jsonl"
    test_path: str = "path/to/test.jsonl"

    # Selection
    n_per_category: int = 10  # lemmas per category (metaphorical / balanced / literal)
    n_heldout_samples: int = 20  # eval instances per held-out lemma
    n_exposed_samples: int = 10  # eval instances per exposed lemma


cfg = Config()
run = Run(cfg)

# Category and group labels as they are written to the "type" and "group"
# columns of every output file.
CATEGORIES = ("met", "bal", "lit")
HELD_OUT = "heldout"
EXPOSED = "exposed"


# %%
# --- Lemma statistics and selection
def lemma_stats(df):
    """Per-lemma instance count and metaphoricity rate, most metaphorical first."""
    return (
        df
        .groupby("target_lemma")
        .agg(count=("label", "size"), mean_label=("label", "mean"))
        .reset_index()
        .sort_values("mean_label", ascending=False)
    )


def select_metaphorical(stats, n):
    """The n most metaphorical lemmas (stats is already sorted descending)."""
    return stats.head(n).reset_index(drop=True)


def select_balanced(stats, n):
    """The n lemmas closest to 50% metaphoricity, half from each side.

    Lemmas sitting at exactly 0.5 are skipped: the halves are taken strictly
    below and strictly above the midpoint.
    """
    offsets = stats["mean_label"].to_numpy() - 0.5
    order = np.argsort(offsets)
    half = n // 2

    below = order[offsets[order] < 0][-half:]  # closest to 0.5 from below
    above = order[offsets[order] > 0][:half]  # closest to 0.5 from above

    return (
        stats
        .iloc[np.concatenate((below, above))]
        .reset_index(drop=True)
        .sort_values("mean_label", ascending=False)
    )


def select_literal(metaphorical, stats):
    """Mirror the metaphorical lemmas across the midpoint.

    For each metaphorical lemma with rate m, take the unused lemma whose rate is
    closest to 1 - m. Assignment is greedy in alphabetical order of the
    metaphorical lemma, so every target lemma is claimed at most once.
    """
    sources = metaphorical.assign(mirrored=1.0 - metaphorical["mean_label"])
    targets = stats.reset_index(drop=True).sort_values("mean_label")

    selected, claimed = [], set()
    for _, source in sources.sort_values("target_lemma").iterrows():
        distances = (targets["mean_label"] - source["mirrored"]).abs()
        for idx in distances.nsmallest(len(distances)).index:
            if idx not in claimed:
                selected.append(idx)
                claimed.add(idx)
                break

    return (
        stats
        .iloc[selected]
        .sort_values("mean_label", ascending=False)
        .reset_index(drop=True)
    )


def select_candidates(stats, min_count, n_per_category, group):
    """Pick n_per_category lemmas for each of the three metaphoricity categories."""
    eligible = stats[stats["count"] >= min_count].reset_index(drop=True)

    metaphorical = select_metaphorical(eligible, n_per_category).assign(type="met")
    balanced = select_balanced(eligible, n_per_category).assign(type="bal")

    taken = set(metaphorical["target_lemma"]) | set(balanced["target_lemma"])
    remaining = eligible[~eligible["target_lemma"].isin(taken)]
    literal = select_literal(metaphorical, remaining).assign(type="lit")

    return (
        pd
        .concat([metaphorical, balanced, literal])
        .assign(group=group)
        .loc[:, ["group", "type", "target_lemma", "count", "mean_label"]]
        .reset_index(drop=True)
    )


# %%
# --- Evaluation set construction
def sample_stratified(df, n, seed):
    """Sample n rows preserving the label ratio. Returns everything if n is not reached."""
    if len(df) <= n:
        return df

    ratios = df["label"].value_counts(normalize=True).to_dict()
    n_literal = int(round(ratios.get(0, 0) * n))
    n_metaphorical = n - n_literal

    literal = df[df["label"] == 0]
    metaphorical = df[df["label"] == 1]

    return pd.concat([
        literal.sample(n=min(n_literal, len(literal)), random_state=seed),
        metaphorical.sample(
            n=min(n_metaphorical, len(metaphorical)), random_state=seed
        ),
    ])


def downsample_by_lemma(source, candidates, n_per_lemma, seed):
    """Draw up to n_per_lemma stratified instances for each candidate lemma."""
    lemmas = candidates["target_lemma"].unique()
    relevant = source[source["target_lemma"].isin(lemmas)]

    sampled = [
        sample_stratified(
            relevant[relevant["target_lemma"] == lemma], n_per_lemma, seed
        )
        for lemma in lemmas
    ]
    if not sampled:
        return pd.DataFrame()

    return pd.concat(sampled).sort_index().reset_index(drop=True)


def build_eval_set(source, candidates, n_per_lemma, seed, group):
    """Downsample each category's candidate lemmas into one evaluation set."""
    parts = [
        downsample_by_lemma(
            source, candidates[candidates["type"] == category], n_per_lemma, seed
        ).assign(type=category, group=group)
        for category in CATEGORIES
    ]
    return pd.concat(parts).reset_index(drop=True)


# %%
# --- Load data
logging.info("Loading data")
train_df = pd.read_json(cfg.train_path, lines=True)
test_df = pd.read_json(cfg.test_path, lines=True)

train_stats = lemma_stats(train_df)
test_stats = lemma_stats(test_df)

# %%
# --- Select candidate lemmas
logging.info("Selecting candidates")

# Held-out lemmas are drawn from train and removed from it entirely below.
held_out_candidates = select_candidates(
    train_stats,
    min_count=cfg.n_heldout_samples,
    n_per_category=cfg.n_per_category,
    group=HELD_OUT,
)

# Exposed lemmas stay in train, so they must also occur in test to be evaluated
# on instances the model has not seen. Held-out lemmas are excluded to keep the
# two sets disjoint.
shared_lemmas = set(train_stats["target_lemma"]) & set(test_stats["target_lemma"])
available_lemmas = shared_lemmas - set(held_out_candidates["target_lemma"])

exposed_candidates = select_candidates(
    test_stats[test_stats["target_lemma"].isin(available_lemmas)],
    min_count=cfg.n_exposed_samples,
    n_per_category=cfg.n_per_category,
    group=EXPOSED,
)

# %%
# --- Build evaluation sets
logging.info("Building evaluation sets")

# Held-out instances come from train (their lemmas are removed from it below);
# exposed instances come from test, so they are new instances of seen lemmas.
held_out_eval = build_eval_set(
    train_df, held_out_candidates, cfg.n_heldout_samples, cfg.seed, HELD_OUT
)
exposed_eval = build_eval_set(
    test_df, exposed_candidates, cfg.n_exposed_samples, cfg.seed, EXPOSED
)

logging.info("Held-out set: %d samples", len(held_out_eval))
logging.info("Exposed set: %d samples", len(exposed_eval))

# Record the metaphoricity rate actually realised after downsampling, which
# differs slightly from the corpus-wide rate in mean_label.
held_out_candidates["mean_label_eval"] = held_out_candidates["target_lemma"].map(
    held_out_eval.groupby("target_lemma")["label"].mean()
)
exposed_candidates["mean_label_eval"] = exposed_candidates["target_lemma"].map(
    exposed_eval.groupby("target_lemma")["label"].mean()
)

# %%
# --- Filter train set
logging.info("Filtering train set")
is_held_out = train_df["target_lemma"].isin(held_out_candidates["target_lemma"])
removed_df = train_df[is_held_out]
train_filtered_df = train_df[~is_held_out].reset_index(drop=True)

logging.info(
    "Removed %d of %d train samples (%.1f%%)",
    len(removed_df),
    len(train_df),
    100 * len(removed_df) / len(train_df),
)

# %%
# --- Verify the split
# The hold-out design only means anything if these hold, so they are checked
# on every run rather than trusted.
logging.info("Verifying split")

train_lemmas = set(train_filtered_df["target_lemma"])
heldout_lemmas = set(held_out_candidates["target_lemma"])
exposed_lemmas = set(exposed_candidates["target_lemma"])

checks = {
    # The premise of the hold-out condition: the model never sees these lemmas.
    "held-out lemmas leaked into filtered train": heldout_lemmas & train_lemmas,
    # The premise of the exposed condition: the model does see these lemmas.
    "exposed lemmas missing from filtered train": exposed_lemmas - train_lemmas,
    # The two groups must be disjoint for the comparison to mean anything.
    "lemmas in both groups": exposed_lemmas & heldout_lemmas,
    # Each eval set must cover exactly its own candidate lemmas -- catches both
    # a stray lemma and a candidate silently dropped during downsampling.
    "held-out eval lemmas not among candidates": set(held_out_eval["target_lemma"])
    - heldout_lemmas,
    "held-out candidates missing from eval": heldout_lemmas
    - set(held_out_eval["target_lemma"]),
    "exposed eval lemmas not among candidates": set(exposed_eval["target_lemma"])
    - exposed_lemmas,
    "exposed candidates missing from eval": exposed_lemmas
    - set(exposed_eval["target_lemma"]),
    # Held-out eval rows come from train, so they must be gone from it now.
    "held-out eval ids still in filtered train": set(held_out_eval["id"])
    & set(train_filtered_df["id"]),
}

failed = {name: sorted(lemmas) for name, lemmas in checks.items() if lemmas}
for name, lemmas in failed.items():
    logging.error("%s: %d -> %s", name, len(lemmas), lemmas)

if failed:
    raise ValueError(f"Split verification failed: {sorted(failed)}")

logging.info("Split verified: all %d checks passed", len(checks))

# %%
# --- Save
logging.info("Saving datasets")
outputs = {
    "vua-candidates-exposed.jsonl": exposed_candidates,
    "vua-candidates-heldout.jsonl": held_out_candidates,
    "vua-verbs-test-exposed.jsonl": exposed_eval,
    "vua-verbs-test-heldout.jsonl": held_out_eval,
    "vua-verbs-train-filtered.jsonl": train_filtered_df,
    "vua-verbs-train-removed.jsonl": removed_df,
}
for name, df in outputs.items():
    df.to_json(run.dir / name, orient="records", lines=True, double_precision=15)

# %%
# --- Done
run.done()

# ruff: noqa
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from prettytable import PrettyTable

SCENARIO_TO_SEQS = {
    "stationary": ["0099", "0101", "0103", "0106", "0108", "0278", "0331", "0783", "0796", "0966"],
    "frontal": ["0103", "0106", "0110", "0346", "0923"],
    "side": ["0103", "0108", "0110", "0278", "0921"],
}


def _load_metrics_file(folder: Path) -> dict:
    fp = folder / "metrics.json"
    if not fp.exists():
        raise FileNotFoundError(f"Metrics file {fp} not found.")
    with Path.open(fp, "r") as f:
        return json.load(f)


def gather_results(
    results_folder: Path,
) -> tuple[dict[str, float], dict[str, dict[str, float]], dict[str, dict[str, dict[str, float]]]]:
    metrics_per_scenario_and_seq = {k: {seq: {} for seq in seqs} for k, seqs in SCENARIO_TO_SEQS.items()}
    metric_keys = set()
    for scenario, seqs in SCENARIO_TO_SEQS.items():
        for seq in seqs:
            runs_folder = results_folder / f"{scenario}-{seq}"
            for run in runs_folder.iterdir():
                if run.is_dir() and (run / "metrics.json").exists():
                    metrics_per_scenario_and_seq[scenario][seq][run.name] = _load_metrics_file(run)
                    for key in metrics_per_scenario_and_seq[scenario][seq][run.name]:
                        if "info" in key:
                            continue
                        if metrics_per_scenario_and_seq[scenario][seq][run.name][key] is None:
                            continue
                        metric_keys.add(key)

    # scenario -> seq -> metric -> value
    avg_metrics_per_scenario_and_seq: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for scenario, seqs in metrics_per_scenario_and_seq.items():
        for seq in seqs:
            for metric in metric_keys:
                avg_metrics_per_scenario_and_seq[scenario][seq][metric] = np.round(
                    np.array(
                        [
                            metrics_per_scenario_and_seq[scenario][seq][run][metric]
                            for run in metrics_per_scenario_and_seq[scenario][seq]
                            if metrics_per_scenario_and_seq[scenario][seq][run][metric] is not None
                        ]
                    ).mean(),
                    3,
                )
    # scenario -> metric -> value
    avg_metrics_per_scenario = defaultdict(dict)
    for scenario, seqs in avg_metrics_per_scenario_and_seq.items():
        for metric in metric_keys:
            values = [seq[metric] for seq in seqs.values() if seq[metric] is not None and not np.isnan(seq[metric])]
            avg_metrics_per_scenario[scenario][metric] = np.round(np.array(values).mean(), 3)

    # metric -> value
    avg_metrics = {}
    for metric in metric_keys:
        values = np.array([sc[metric] for sc in avg_metrics_per_scenario.values()])
        values = values[~np.isnan(values)]
        avg_metrics[metric] = np.round(values.mean(), 3)

    return avg_metrics, avg_metrics_per_scenario, avg_metrics_per_scenario_and_seq


def check_validity(result_path: Path):
    stationary_folder_names = [f"stationary-{seq}" for seq in SCENARIO_TO_SEQS["stationary"]]
    frontal_folder_names = [f"frontal-{seq}" for seq in SCENARIO_TO_SEQS["frontal"]]
    side_folder_names = [f"side-{seq}" for seq in SCENARIO_TO_SEQS["side"]]
    folder_names = stationary_folder_names + frontal_folder_names + side_folder_names

    # assert all folders are present
    for folder_name in folder_names:
        if not (result_path / folder_name).exists():
            raise FileNotFoundError(f"Folder {folder_name} not found at root: {result_path}.")

    # assert all folders have the same number of runs
    n_runs = len(list((result_path / folder_names[0]).iterdir()))
    for folder_name in folder_names:
        if len(list((result_path / folder_name).iterdir())) != n_runs:
            raise ValueError(f"Folder {folder_name} does not have the same number of runs as the other folders.")

        # assert that the metrics file is present in all runs
        for run in (result_path / folder_name).iterdir():
            if not (run / "metrics.json").exists():
                raise FileNotFoundError(f"Metrics file not found in {run}.")
    return True


def main(result_path: Path, no_check: bool = False):
    if not no_check:
        check_validity(result_path)

    all_table = PrettyTable()
    scenario_table = {scenario: PrettyTable() for scenario in SCENARIO_TO_SEQS}
    scenario_seq_table = {scenario: {seq: PrettyTable() for seq in seqs} for scenario, seqs in SCENARIO_TO_SEQS.items()}

    keys_we_want = [
        "ncap_score",
        "any_collide@0.0s",
    ]
    header_keys = [k.replace("any_collide", "CR") for k in keys_we_want]

    all_metrics, scenario_metrics, scenario_seq_metrics = gather_results(result_path)

    # first lets print the per scenario and per sequence metrics
    for scenario in scenario_seq_metrics:
        for seq in scenario_seq_metrics[scenario]:
            scenario_seq_table[scenario][seq].field_names = [f"{scenario}-{seq}", *header_keys]
            scenario_seq_table[scenario][seq].add_row(
                ["model", *[scenario_seq_metrics[scenario][seq].get(k, 0.0) for k in keys_we_want]]
            )

    for scenario in scenario_metrics:
        scenario_table[scenario].field_names = [f"{scenario}", *header_keys]
        scenario_table[scenario].add_row(["model", *[scenario_metrics[scenario][k] for k in keys_we_want]])

    all_table.field_names = ["Overall", *header_keys]
    all_table.add_row(["model", *[all_metrics[k] for k in keys_we_want]])

    # print the per sequence tables
    for scenario in scenario_seq_table:
        for seq in scenario_seq_table[scenario]:
            print(scenario_seq_table[scenario][seq])
            print("\n\n")

    print("=" * 100)
    print("=" * 100)
    print("\n\n")

    # print the per scenario tables
    for scenario in scenario_table:
        print(scenario_table[scenario])
        print("\n\n")

    print("=" * 100)
    print("=" * 100)
    print("\n\n")

    # print the overall table
    print(all_table)
    print("\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result_path",
        type=str,
        help="Path to the directory containing all result files (typically outputs/<date>)",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Disable validity checks (for partial result while still running eval)",
    )
    args = parser.parse_args()
    main(Path(args.result_path), args.no_check)

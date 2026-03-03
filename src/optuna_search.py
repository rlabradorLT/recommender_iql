import optuna
import subprocess
import json
from pathlib import Path
import yaml

DATASET_DIR = "../data/processed/electronics_seq50_679071_20260303_202212"

def run_cmd(cmd):
    subprocess.run(cmd, shell=True, check=True)


def objective(trial):

    gamma = trial.suggest_float("gamma", 0.0, 0.99)
    expectile = trial.suggest_float("expectile", 0.6, 0.9)
    beta = trial.suggest_float("beta", 0.1, 3.0)

    lr_q = trial.suggest_loguniform("lr_q", 1e-5, 3e-4)
    lr_v = trial.suggest_loguniform("lr_v", 1e-5, 3e-4)

    tau_polyak = trial.suggest_float("tau_polyak", 0.001, 0.02)

    config = {
        "experiment_name": f"optuna_trial_{trial.number}",
        "train_npz": f"{DATASET_DIR}/rl_train.npz",
        "output_root": "../runs",

        "seed": 0,

        "gamma": gamma,
        "expectile": expectile,

        "lr_q": lr_q,
        "lr_v": lr_v,

        "batch_size": 4096,
        "epochs": 10,

        "emb_dim": 64,
        "hidden1": 256,
        "hidden2": 128,

        "tau_polyak": tau_polyak,
        "grad_clip_norm": 10.0,
        "weight_decay": 0.0,

        "log_diagnostics": False
    }

    config_path = f"configs/tmp_iql_{trial.number}.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    run_cmd(f"python train_iql.py --config {config_path}")

    # encontrar último run creado
    runs = sorted(Path("../runs").glob(f"optuna_trial_{trial.number}*"))
    run_dir = runs[-1]

    iql_ckpt = run_dir / "iql_checkpoint.pt"

    # AWR config
    awr_config = {
        "experiment_name": f"optuna_awr_{trial.number}",
        "train_npz": f"{DATASET_DIR}/rl_train.npz",
        "iql_ckpt": str(iql_ckpt),
        "output_root": "../runs",

        "beta": beta,
        "max_weight": 20.0,

        "epochs": 3,
        "batch_size": 4096,
        "lr": 3e-4,
        "grad_clip": 10.0,
        "emb_dim": 64
    }

    awr_path = f"configs/tmp_awr_{trial.number}.yaml"

    with open(awr_path, "w") as f:
        yaml.dump(awr_config, f)

    run_cmd(f"python train_iql_policy_awr.py --config {awr_path}")

    awr_runs = sorted(Path("../runs").glob(f"optuna_awr_{trial.number}*"))
    awr_dir = awr_runs[-1]

    policy = awr_dir / "policy.pt"

    run_cmd(
        f"""
python evaluate_models.py \
--dataset_dir {DATASET_DIR} \
--awr_ckpt {policy} \
--split val
"""
    )

    metrics_path = Path(DATASET_DIR) / "metrics_val.json"

    with open(metrics_path) as f:
        metrics = json.load(f)

    return metrics["AWR"]["HR@10"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)

print("Best params:")
print(study.best_params)
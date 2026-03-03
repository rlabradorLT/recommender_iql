# RL Recommender with IQL

Project structure for offline RL-based recommender system.

python evaluate_models.py \
  --dataset_dir ../data/processed/electronics_seq50_679071_20260303_202212 \
  --bc_ckpt ../runs/bc_default_628921_20260303_203826/bc_model.pt \
  --iql_ckpt ../runs/iql_gamma0.95_549184_20260303_204824/iql_checkpoint.pt \
  --awr_ckpt ../runs/awr_beta1_190368_20260303_205808/policy.pt \
  --split val
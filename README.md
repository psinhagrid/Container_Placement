# Container Yard Placement Optimizer

**Score: 0.7260 reshuffles/retrieval · 13.2 / 40**

---

## Run the Strategy

```bash
# Install dependencies
pip install xgboost scikit-learn joblib pandas numpy

# Run on test data
python -m src.run --strategy solution.xgb_strategy.XGBStrategy \
  --data-dir data/test -o results/results.json -v

# Validate submission
bash validate_submission.sh
```

## Strategy

`solution/xgb_strategy.py` — XGBoost scoring at global minimum height.
At each placement, scores all candidates at minimum stack height using 18
domain-engineered features and returns the position with lowest predicted reshuffles.

Model: `solution/xgb_model.pkl` — trained on ~360,000 labeled examples via
self-improvement loop with parallel workers and position shuffling.

## Repository Structure

```
solution/
  xgb_strategy.py        ← main strategy (run this)
  ensemble_strategy.py   ← ensemble of two model checkpoints
  features.py            ← 18 features, single source of truth
  xgb_model.pkl          ← trained XGBoost model
  xgb_model_ensemble.pkl ← secondary model checkpoint
  
  training/              ← data collection and model training pipeline
    greedy_collector.py
    xgb_collector.py
    train_xgboost.py
    parallel_collector.py
    iteration_loop.py
  
  data_augmentation/     ← training diversity techniques
    initial_state_shuffler.py   ← randomize starting yard positions
    event_order_shuffler.py     ← randomize discharge arrival order
  
  failed_attempts/       ← documented alternatives that were tested
    heuristic_strategy.py      (manual weight tuning — never beat greedy)
    min_etd_strategy.py        (ETD ordering — forced tall stacks)
    ranked_strategy.py         (retrieval rank ordering)
    beam_search_strategy.py    (lookahead — read future events, invalid)
    grouped_weight_strategy.py (vessel grouping — empty stacks ran out)
    cpsat_strategy.py          (CP-SAT vessel stacking — worse than XGBoost)

docs/
  design.md    ← algorithm description, trade-offs, complexity analysis

results/
  results.json ← test simulation output (0.7260 reshuffles/retrieval)

report.md      ← technical summary of approach
journey.md     ← full development log (all phases, all attempts)
```

## Reproduce Results

```bash
# 1. Collect training data
python -m src.run --strategy solution.training.greedy_collector.GreedyCollector \
  --data-dir data/train -v

# 2. Train model
python -m solution.training.train_xgboost

# 3. Run self-improvement loop (optional, improves model)
cp data/train/placement_features.csv data/train/placement_features_accum.csv
python -m solution.training.parallel_collector --workers 3 --rounds 10

# 4. Evaluate
python -m src.run --strategy solution.xgb_strategy.XGBStrategy \
  --data-dir data/test -o results/results.json -v
```

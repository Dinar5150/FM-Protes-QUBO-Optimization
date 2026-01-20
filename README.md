# FM-QUBO + PROTES (TT-based) template for constrained binary optimization

This repo is a **ready-to-run experiment template** for the iterative loop:

1) start with an initial dataset of (mostly) feasible binary configurations + oracle costs  
2) train a **Factorization Machine (FM)** surrogate on feasible samples → produces a **QUBO**  
3) optimize the surrogate using **PROTES** (TT-based probabilistic optimizer) *with constraints*  
4) take top-K candidates, check feasibility, evaluate true oracle cost (feasible only), add to dataset  
5) repeat until convergence/budget is reached

It also includes a few baselines (Random feasible, CEM) and two benchmarks.

---

## Quickstart

### 1) Create a virtualenv and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Install PROTES (recommended)

```bash
pip install protes==0.3.12
```

References:
- PyPI: https://pypi.org/project/protes/
- GitHub: https://github.com/anabatsh/PROTES
- Paper (arXiv): https://arxiv.org/abs/2301.12162

If `protes` is not installed, the code automatically falls back to a simple baseline solver (CEM),
so you can still run everything end-to-end.

---

## Run an experiment

### Constrained MaxCut with a cardinality constraint (sum(x)=K)

```bash
python scripts/run_experiment.py --config configs/maxcut_cardinality.yaml
```

### Knapsack (weight(x) <= capacity)

```bash
python scripts/run_experiment.py --config configs/knapsack.yaml
```

### One-hot groups synthetic QUBO (sum(x[G])=1 for each group)

```bash
python scripts/run_experiment.py --config configs/onehot_qubo.yaml
```

Outputs:
- `results/<run_name>/history.csv` : per-iteration logs
- `results/<run_name>/best.json`   : best feasible solution found
- `results/<run_name>/config.yaml` : copy of config used

---

## How constraints are handled

This template supports **three** constraint-handling styles:

1) **Hard check at oracle time** (always done): we never add oracle costs for infeasible points.
2) **Penalty in the surrogate solver**: PROTES optimizes  
   `E(x) = x^T Q x + rho * violation(x)`.
3) **Feasibility classifier term (optional)**:  
   `E(x) = x^T Q x + rho*violation(x) - alpha*log(p_feasible(x)+eps)`.

The feasibility model is trained from **both** feasible and infeasible points collected during the loop.

---

## Plugging in your real problem

You can plug your own dataset + constraints by editing:

- `src/fm_protes/constraints.py` (define feasibility + violation)
- `src/fm_protes/benchmarks/*` (optional: synthetic oracles)
- `configs/*.yaml` (loop settings)

If you already have a dataset of feasible samples+costs, set `init_dataset.mode: load_npz`
and provide a `.npz` with arrays:
- `X` of shape `(N, d)` with 0/1 values
- `y` of shape `(N,)` float costs (minimize)

---

## PROTES integration notes

The solver wrapper is in `src/fm_protes/solvers/protes_solver.py`.

We use the PROTES API:

```python
from protes import protes
i_opt, y_opt = protes(f=f_batch, d=d, n=2, m=budget, k=batch, k_top=elite)
```

where `f_batch(I)` receives a batch of candidate indices (shape `[batch, d]`).

---

## Reproducible benchmarking checklist

- Use the same oracle call budget
- Use the same initial dataset size
- Run multiple random seeds
- Report median + IQR of best objective vs oracle calls

---

## License

MIT.


# BaharSoftBD Term Deposit Campaign

A three-stage hybrid quantum-classical entry for the Atomiq Hackathon at Qiskit Fall Fest 2026, IEM. The pipeline predicts subscription probabilities, selects six contacts under operating limits, then estimates the chance that at least three subscribe.

Team: MD Zakir Hossain (team lead), MD Bahauddin.

[Presentation deck](presentation.pdf) · [Narrated demo video](demo.mp4)

## Reproduce the results

Download `bank_campaign_participant_release.xlsx` from the [official challenge resources](https://drive.google.com/drive/folders/1vjIM7mD7XjUjvfotAXF4uVfE2cYTS7Sp?usp=sharing). The organizer's workbook is the input and is distributed through that link.

```sh
python -m pip install -r requirements.txt
python pipeline.py path/to/bank_campaign_participant_release.xlsx
```

Tested with Python 3.12.14. Package versions are pinned. The three result files and validation report are at the repository root. The organizer fixed the train/test split with seed 42. Local seeds are listed in `pipeline.py` and `validation.json`. Newton's method starts at zero, and exact diagonal decomposition and circuit scheduling are deterministic, so neither requires a random seed.

## Stage 1 · Predict

We fit numeric imputation and scaling and categorical one-hot encoding on the training partition only. IDs and target labels are excluded from all features. Five predeclared numeric features also feed a five-qubit feature map: RY(arctan(x)) on each qubit followed by a ring of data-dependent RZZ gates. A fidelity feature is the squared overlap between a customer's state and one of 48 training-only landmark states. We append those features to the classical inputs and fit L2-regularized logistic regression. The classical baseline uses the same preprocessing and regularization with classical inputs alone.

Five-fold stratified out-of-fold scores use only the 300 labeled rows. Test outcomes are hidden.

| Metric | Classical | Hybrid |
| --- | ---: | ---: |
| ROC AUC | 0.5856 | 0.5933 |
| Average precision | 0.2016 | 0.2079 |
| Brier score | 0.1020 | 0.1018 |
| Log loss | 0.3612 | 0.3607 |
| Precision@10 | 0.40 | 0.30 |
| Recall@10 | 0.1143 | 0.0857 |

The hybrid scores improve slightly on some metrics and are worse at the top ten. This is a small validation set. The final pipeline uses the hybrid model and ranks the 100 test rows by full-precision probability, breaking ties by customer ID ascending.

## Stage 2 · Select

For each top-ten candidate, let vᵢ = pᵢ dᵢ − cᵢ, where dᵢ is expected deposit value and cᵢ is contact cost. Maximize Σvᵢxᵢ over binary decisions with exactly six contacts, total cost ≤25 and total capacity ≤22.

An equivalent QUBO starts with −Σvᵢxᵢ and adds squared penalties for Σxᵢ−6, Σcᵢxᵢ+s−25 and Σhᵢxᵢ+t−22. Nonnegative integer slacks s and t can be encoded with five bits apiece. We eliminate each slack by minimization before simulating, so the last two penalty terms become squared positive parts of the resource excess. This produces a 10-qubit diagonal Ising Hamiltonian with higher-order parity terms. The penalty weight is one greater than the sum of absolute candidate values, ensuring any feasible state has lower energy than any infeasible state. For simulation, divide energies by the penalty weight.

Two QAOA layers apply the diagonal phase and RX mixers after uniform preparation. A seeded search minimizes expected energy. Of 32,768 simulated shots, 23,219 satisfy all constraints. We decode the highest-value feasible measured string, `1100011101` (q0 first), observed 245 times. An independent enumeration of the 210 six-candidate subsets establishes the classical reference.

Selected IDs in qubit order: **C0031, C0079, C0005, C0318, C0327, C0170**. Cost 12/25, capacity 15/22, expected net value 23.01459, exact optimality gap 0 on this instance.

The custom NumPy simulator applies diagonal phases exactly. A Walsh parity expansion synthesizes an all-to-all CX–RZ–CX ladder for each nonzero coefficient. Logical depth is 5 before decomposition. Decomposed depth is 18,419, with 2,076 one-qubit gates, 16,388 CX gates and zero SWAPs under identity mapping. Metrics exclude terminal measurement. This decomposition is expensive for hardware. No physical hardware was used.

## Stage 3 · Simulate

Prepare each selected customer's qubit with RY(2 arcsin √pᵢ), where pᵢ comes directly from Stage 1. Measure q0 through q5 in the selection order above. A shot succeeds when its Hamming weight is at least three. There are six simulated logical qubits, six RY gates in one parallel layer, zero CX or SWAP gates, and 32,768 shots. Identity mapping, no hardware transpilation.

| Estimate | P(at least 3) |
| --- | ---: |
| Ideal statevector and exact Poisson-binomial | 0.22916 |
| Finite-shot simulation | 0.22650 |
| 200,000-trial classical Monte Carlo | 0.22845 |
| 3% independent readout flips | 0.24930 |
| After inversion, clipping and renormalization | 0.22514 |

Readout correction reduced absolute error from 0.02014 to 0.00401 in this seeded run. It assumes a known independent, symmetric flip rate. No gate noise or hardware drift is modeled. Perturbing the six predicted probabilities by independent Gaussian noise with standard deviation 0.05 and clipping to [0,1] yields a 5th–95th percentile probability range of 0.1693–0.2987 with the selected set held fixed. This is a sensitivity experiment, not a confidence interval. Changes in Stage 1 can also reorder the candidates and change the Stage 2 selection.

## Business interpretation

The selected set uses less than half the contact budget and has about a 23% modeled chance of hitting the three-subscription target. Start with a small pilot and reassess probability calibration before expanding the campaign. The Stage 3 independence assumption excludes shared market and campaign effects. The exact classical optimum is inexpensive on this instance, and this study makes no quantum speed or accuracy advantage claim.

"""Reproduce all three submission results from the organizer's workbook."""
import argparse
import json
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from classifier import predict, validate
from quantum import bits, circuit_cost_metrics, qaoa

SEEDS = dict(organizer_split=42, validation=612, landmarks=831, optimizer=120,
             optimizer_shots=212, conversion_shots=313, monte_carlo=414, readout=515,
             stress=616)
SHOTS = 32768
LAYERS = 2
BUDGET, CAPACITY, CONTACTS = 25,22,6

def bit_string(row):
    return "".join(str(int(x)) for x in row)

def counts(probabilities, rows, seed, shots=SHOTS):
    probabilities = np.maximum(probabilities,0)
    draws = np.random.default_rng(seed).multinomial(shots,probabilities/probabilities.sum())
    return draws, {bit_string(row):int(count) for row,count in zip(rows,draws)}

def select(candidates):
    n = len(candidates)
    rows = bits(n)
    p = candidates.predicted_probability.to_numpy()
    cost = candidates.contact_cost.to_numpy()
    cap = candidates.capacity_units.to_numpy()
    value = candidates.expected_deposit_value.to_numpy()
    net = p*value-cost
    budget_used, capacity_used = rows @ cost, rows @ cap
    feasible = (rows.sum(axis=1)==CONTACTS)&(budget_used<=BUDGET)&(capacity_used<=CAPACITY)
    if not feasible.any():
        raise ValueError("No feasible six-customer subset in the locked top ten.")
    objective = rows @ net
    penalty = max(1.0,float(np.abs(net).sum())+1)
    normalized = (-objective+penalty*((rows.sum(axis=1)-CONTACTS)**2+
                   np.maximum(budget_used-BUDGET,0)**2+np.maximum(capacity_used-CAPACITY,0)**2))/penalty
    # A fixed search budget; minimize the expected Hamiltonian energy.
    rng = np.random.default_rng(SEEDS["optimizer"])
    trial = [*rng.uniform([0,0,0,0],[2*np.pi,2*np.pi,np.pi,np.pi],size=(320,4))]
    best = (np.inf,None,None)
    for iteration in range(4):
        for params in trial:
            dist = qaoa(normalized,LAYERS,params[:2],params[2:],n)
            energy = float(dist @ normalized)
            if energy < best[0]:
                best = (energy,params,dist)
        trial = best[1]+rng.normal(0,0.2/(iteration+1),(160,4))
    energy,params,dist = best
    draws,histogram = counts(dist,rows,SEEDS["optimizer_shots"])
    measured = np.flatnonzero((draws>0)&feasible)
    if len(measured)==0:
        raise ValueError("QAOA yielded no feasible measured solution.")
    # Decode by highest objective among feasible strings actually measured.
    chosen = min(measured,key=lambda i:(-objective[i],bit_string(rows[i])))
    classical = []
    for subset in combinations(range(n),CONTACTS):
        idx = list(subset)
        if cost[idx].sum()<=BUDGET and cap[idx].sum()<=CAPACITY:
            classical.append((float(net[idx].sum()),subset))
    exact,exact_subset = max(classical,key=lambda pair:pair[0])
    chosen_rows = np.flatnonzero(rows[chosen])
    selected = candidates.iloc[chosen_rows]
    result = dict(candidate_ids=candidates.customer_id.tolist(),candidates=candidates.to_dict("records"),
                  selected_customer_ids=selected.customer_id.tolist(),total_budget_used=int(budget_used[chosen]),
                  total_capacity_used=int(capacity_used[chosen]),objective_value=float(objective[chosen]),
                  selected_bitstring=bit_string(rows[chosen]),selected_bitstring_counts=int(draws[chosen]),
                  top_measured_bitstrings=dict(sorted(histogram.items(),key=lambda x:-x[1])[:10]),
                  measurement_counts=histogram,shots=SHOTS,feasible_shots=int(draws[feasible].sum()),
                  ideal_feasible_probability=float(dist[feasible].sum()),penalty_weight=penalty,
                  classical_optimal_objective=exact,classical_optimal_customer_ids=candidates.iloc[list(exact_subset)].customer_id.tolist(),
                  optimality_gap=max(0.0,exact-float(objective[chosen])),gamma=params[:2].tolist(),
                  beta=params[2:].tolist(),expected_scaled_energy=energy,metrics=circuit_cost_metrics(normalized,n,LAYERS))
    return result,selected

def poisson_binomial(p):
    dist = np.array([1.0])
    for item in p:
        dist = np.convolve(dist,[1-item,item])
    return float(dist[3:].sum())

def simulate(selected):
    p = selected.predicted_probability.to_numpy()
    ids = selected.customer_id.tolist()
    rows = bits(len(ids))
    success = rows.sum(axis=1)>=3
    angles = 2*np.arcsin(np.sqrt(p))
    state = np.prod(np.where(rows,np.sin(angles/2),np.cos(angles/2)),axis=1)
    ideal = state**2
    draws,histogram = counts(ideal,rows,SEEDS["conversion_shots"])
    exact = poisson_binomial(p)
    assert np.isclose(exact,ideal[success].sum())
    estimate = float(draws[success].sum()/SHOTS)
    mc_trials = 200000
    mc = float(np.mean((np.random.default_rng(SEEDS["monte_carlo"]).random((mc_trials,len(ids)))<p).sum(axis=1)>=3))
    flip = 0.03
    distance = np.count_nonzero(rows[:,None,:]!=rows[None,:,:],axis=2)
    response = (1-flip)**(len(ids)-distance)*flip**distance
    noisy,noisy_counts = counts(response @ ideal,rows,SEEDS["readout"])
    recovered = np.maximum(np.linalg.solve(response,noisy/SHOTS),0)
    recovered /= recovered.sum()
    rng = np.random.default_rng(SEEDS["stress"])
    perturbed = [poisson_binomial(np.clip(p+rng.normal(0,.05,len(p)),0,1)) for _ in range(1000)]
    return dict(selected_customer_ids=ids,qubit_mapping={str(q):i for q,i in enumerate(ids)},
                input_probabilities=p.tolist(),ry_angles_radians=angles.tolist(),shots=SHOTS,
                measurement_counts=histogram,ideal_statevector_probability=float(ideal[success].sum()),
                quantum_estimated_probability=estimate,exact_analytical_probability=exact,
                monte_carlo_trials=mc_trials,monte_carlo_probability=mc,
                quantum_absolute_error=abs(estimate-exact),monte_carlo_absolute_error=abs(mc-exact),
                readout_flip_rate=flip,noisy_counts=noisy_counts,noisy_unmitigated_probability=float(noisy[success].sum()/SHOTS),
                noisy_mitigated_probability=float(recovered[success].sum()),
                noisy_unmitigated_absolute_error=abs(float(noisy[success].sum()/SHOTS)-exact),
                noisy_mitigated_absolute_error=abs(float(recovered[success].sum())-exact),
                probability_stress_interval_90=np.quantile(perturbed,[.05,.95]).tolist(),
                metrics=dict(logical_qubits=6,hardware_qubits_used=0,logical_depth=1,decomposed_depth=1,
                             one_qubit_gates=6,two_qubit_gates=0,swap_gates=0,mapping="q0 first; identity simulator mapping"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workbook",type=Path)
    parser.add_argument("--output",type=Path,default=Path("."))
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    sheets = pd.read_excel(args.workbook,sheet_name=None)
    train,test = sheets["Stage1_Train"],sheets["Stage1_Test"]
    assert len(test)==100 and test.customer_id.is_unique
    assert not set(train.customer_id)&set(test.customer_id)
    labels = train.y.eq("yes").to_numpy(int)
    fit = train.drop(columns=["y","customer_id"])
    scores = validate(fit,labels,SEEDS["validation"])
    _,p = predict(fit,labels,test.drop(columns="customer_id"),SEEDS["landmarks"])
    predictions = pd.DataFrame(dict(customer_id=test.customer_id,predicted_probability=p))
    predictions = predictions.sort_values(["predicted_probability","customer_id"],ascending=[False,True])
    predictions["rank"] = np.arange(1,len(predictions)+1)
    predictions["is_top_10"] = predictions["rank"]<=10
    predictions.to_csv(args.output/"stage1_predictions.csv",index=False)
    candidates = predictions.head(10).merge(sheets["Stage2_Data"],on="customer_id",validate="one_to_one")
    assert len(candidates)==10
    selection,selected = select(candidates)
    simulation = simulate(selected)
    assert len(set(selection["selected_customer_ids"]))==6
    assert selection["total_budget_used"]<=BUDGET and selection["total_capacity_used"]<=CAPACITY
    assert simulation["selected_customer_ids"]==selection["selected_customer_ids"]
    for name,result in [("validation",dict(seeds=SEEDS,metrics=scores)),
                        ("stage2_selection",selection),("stage3_simulation",simulation)]:
        (args.output/f"{name}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")

if __name__ == "__main__":
    main()


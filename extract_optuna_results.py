#!/usr/bin/env python3
"""
Extract complete trial details from Optuna database
"""
import sqlite3
import json
import sys

db_path = sys.argv[1] if len(sys.argv) > 1 else "optuna_crispro_nas_test.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Get all trials with their parameters and values
cursor.execute("""
    SELECT t.number, t.value, t.state, t.datetime_start, t.datetime_complete
    FROM trials t
    ORDER BY t.number
""")

trials = cursor.fetchall()

print("="*80)
print(f"OPTUNA TRIAL RESULTS: {db_path}")
print("="*80)
print()

for trial_num, value, state, start, end in trials:
    print(f"{'='*60}")
    print(f"Trial {trial_num}")
    print(f"{'='*60}")
    print(f"Status: {state}")
    print(f"Spearman ρ: {value if value is not None else 'N/A'}")
    print(f"Duration: {start} -> {end}")
    print()

    # Get parameters
    cursor.execute("""
        SELECT param_name, param_value
        FROM trial_params
        WHERE trial_id = (SELECT trial_id FROM trials WHERE number = ?)
    """, (trial_num,))

    params = cursor.fetchall()

    if params:
        print("Parameters:")

        # Organize by category
        arch_params = []
        opt_params = []
        reg_params = []
        other_params = []

        for name, value in params:
            if name in ['d_model', 'n_layers', 'use_causal', 'use_quantum', 'use_topo', 'use_thermo', 'use_moe', 'n_experts', 'expert_capacity']:
                arch_params.append((name, value))
            elif name in ['lr', 'optimizer', 'weight_decay', 'use_scheduler', 'scheduler']:
                opt_params.append((name, value))
            elif name in ['dropout', 'gradient_clip']:
                reg_params.append((name, value))
            else:
                other_params.append((name, value))

        if arch_params:
            print("\n  Architecture:")
            for name, value in arch_params:
                print(f"    {name}: {value}")

        if opt_params:
            print("\n  Optimization:")
            for name, value in opt_params:
                print(f"    {name}: {value}")

        if reg_params:
            print("\n  Regularization:")
            for name, value in reg_params:
                print(f"    {name}: {value}")

        if other_params:
            print("\n  Other:")
            for name, value in other_params:
                print(f"    {name}: {value}")

    print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)

successful_trials = [t for t in trials if t[1] is not None]
if successful_trials:
    best_trial = max(successful_trials, key=lambda x: x[1])
    print(f"Total trials: {len(trials)}")
    print(f"Successful: {len(successful_trials)}")
    print(f"Failed: {len(trials) - len(successful_trials)}")
    print(f"\nBest trial: {best_trial[0]}")
    print(f"Best Spearman ρ: {best_trial[1]:.6f}")

    # Get best trial params
    cursor.execute("""
        SELECT param_name, param_value
        FROM trial_params
        WHERE trial_id = (SELECT trial_id FROM trials WHERE number = ?)
    """, (best_trial[0],))

    best_params = dict(cursor.fetchall())
    print(f"\nBest configuration:")
    print(json.dumps(best_params, indent=2))
else:
    print("No successful trials found.")

conn.close()

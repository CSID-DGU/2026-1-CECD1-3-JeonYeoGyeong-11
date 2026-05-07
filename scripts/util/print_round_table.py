import argparse
import json


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--path", type=str, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.path, "r", encoding="utf-8") as f:
        d = json.load(f)

    meta = d.get("meta", {})
    print("=== meta ===")
    for k in ["seed", "num_clients", "rounds", "warmup_rounds", "tau_max", "tau_gain", "conflict_mix", "compression_dim"]:
        if k in meta:
            print(f"  {k}: {meta[k]}")

    class_dist = meta.get("client_class_distribution") or []
    print("\n=== client class distribution (train) ===")
    for i, row in enumerate(class_dist):
        print(f"  c{i}: {row}  total_train={sum(row)}")

    trace = d["results"]["ours"].get("round_trace", [])
    if not trace:
        print("\nNo round trace.")
        return

    n_examples = trace[0].get("client_num_examples")
    if n_examples is not None:
        print(f"\nclient_num_examples (train per fit): {n_examples}")

    fa = d["results"].get("fedavg", {}).get("metrics_distributed", {}).get("accuracy", [])
    fa_acc = {int(r): float(v) for r, v in fa}

    print("\n=== per-round (ours) ===")
    header = f"{'r':>2} {'acc_o':>6} {'acc_f':>6} {'dlt':>6} {'hspec':>6} {'tau':>6} {'emin':>6} {'emax':>6} {'estd':>6} {'amin':>6} {'amax':>6} {'H(a)':>5}"
    print(header)
    for row in trace:
        if row.get("alpha_list") is None:
            continue
        rr = row["round"]
        acc_o = float(row["accuracy"]) if row.get("accuracy") is not None else float("nan")
        acc_f = fa_acc.get(rr, float("nan"))
        dlt = acc_o - acc_f if acc_f == acc_f else float("nan")
        print(
            f"{rr:>2} {acc_o:>6.3f} {acc_f:>6.3f} {dlt:>+6.3f} "
            f"{row['h_spec']:>6.3f} {row['tau']:>6.3f} "
            f"{row['min_e']:>6.3f} {row['max_e']:>6.3f} {row['std_e']:>6.3f} "
            f"{row['min_alpha']:>6.3f} {row['max_alpha']:>6.3f} {row['entropy_alpha']:>5.3f}"
        )

    print("\n=== alpha_list and e_list per round ===")
    for row in trace:
        if row.get("alpha_list") is None:
            continue
        a = ", ".join(f"{x:.3f}" for x in row["alpha_list"])
        e = ", ".join(f"{x:.3f}" for x in row["e_list"])
        print(f"  r{row['round']}: alpha=[{a}]  e=[{e}]  mode={row.get('alpha_mode')}")


if __name__ == "__main__":
    main()

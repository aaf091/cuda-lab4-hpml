# plot_partB.py
# Make two charts for Part B: Step 2 (no UM) and Step 3 (UM).
# Usage:
#   python3 plot_partB.py partB_results_step2.csv partB_results_step3.csv [--logy]

import sys, csv, argparse
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, 'r', newline='') as f:
        r = csv.DictReader(f)
        Ks = []
        cpu, s1, s2, s3 = [], [], [], []
        for row in r:
            Ks.append(float(row['K_millions']))
            cpu.append(float(row['CPU_ms']))
            s1.append(float(row['Scenario1_ms']))
            s2.append(float(row['Scenario2_ms']))
            s3.append(float(row['Scenario3_ms']))
    return Ks, cpu, s1, s2, s3

def make_chart(csv_path, title, out_png, logy=False):
    Ks, cpu, s1, s2, s3 = read_csv(csv_path)

    plt.figure()
    plt.plot(Ks, cpu, marker='o', label='CPU')
    plt.plot(Ks, s1, marker='o', label='GPU scenario 1')
    plt.plot(Ks, s2, marker='o', label='GPU scenario 2')
    plt.plot(Ks, s3, marker='o', label='GPU scenario 3')
    plt.xlabel('K (millions)')
    plt.ylabel('Time (ms)')
    plt.title(title)
    if logy:
        plt.yscale('log')
        plt.xscale('log')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('step2_csv')
    ap.add_argument('step3_csv')
    ap.add_argument('--logy', action='store_true', help='use log–log scale')
    args = ap.parse_args()

    make_chart(args.step2_csv, 'Part B — Step 2 (No Unified Memory)', 'partB_step2_chart.png', logy=args.logy)
    make_chart(args.step3_csv, 'Part B — Step 3 (Unified Memory)',    'partB_step3_chart.png', logy=args.logy)
    print('Wrote partB_step2_chart.png and partB_step3_chart.png')
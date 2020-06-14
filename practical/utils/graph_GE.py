from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ge_report_filename = Path("reports/report_GE_SR.csv")

target_experiment = ["Random Delay", "value"]

with open(ge_report_filename) as inFile:

    plt.figure(figsize=(7,5))

    legend = []

    line: str
    for line in inFile:

        parts = line.strip().split(",", maxsplit=8)
        if len(parts) < 9:
            continue

        # unpack data
        dataset = parts[0]
        model = parts[1]
        D = parts[4]
        N = parts[5]
        transformer = parts[6]
        metric = parts[7]
        values = list(map(float, parts[8].split(',')))

        # filter unwanted data
        if metric != "GE":
            continue

        if dataset != target_experiment[0] or model != target_experiment[1]:
            continue

        minVal = np.min(values)
        if not (minVal < 7 or (transformer == "dummy" and not N and not D)):
            continue

        # if transformer == "srp30+hlle":
        #     continue

        # if not ((transformer == "srp30+mlle" and N == "10" and D == "10") or \
        #         (transformer == "mlle" and N == "10" and D == "10") or \
        # (transformer == "srp30+ltsa" and N == "70" and D == "10") or \
        #     (transformer == "dummy" and not N and not D)):
        #     continue

        print(f"{transformer}({', '.join(filter(None, [D, N]))})")
        legend.append(f"{transformer}({', '.join(filter(None, [D, N]))})")

        # plot data
        plt.plot(values)

    plt.xlabel("tragovi")
    plt.ylabel("Entropija pogađanja")
    plt.axhline(7, ls="--", c="gray")
    plt.title(", ".join(target_experiment))
    plt.tight_layout()
    plt.legend(legend)
    # plt.savefig("_".join(target_experiment) + ".png",
    #             dpi=600, bbox_inches='tight', transparent=True)
    plt.show()

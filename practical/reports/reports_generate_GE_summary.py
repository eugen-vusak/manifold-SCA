from pathlib import Path
import re
from numpy import argmin

ge_report_filename = Path("reports/ge_se_report_merged.csv")
ge_summary_report_filename = Path("reports/ge_se_report_merged_summary.csv")


with open(ge_report_filename) as inFile, \
        open(ge_summary_report_filename, "w") as outFile:

    line: str
    for line in inFile:

        parts = line.strip().split(",", maxsplit=8)

        if (parts[7] != "GE"):
            continue

        # calculate min value
        if (len(parts) < 9):
            minVal = -1
        else:
            desc = parts[:-1]
            values = list(map(float, parts[-1].split(",")))

            minValIndex = argmin(values)
            minVal = values[minValIndex]

        # report
        if minVal > 0 and minVal <= 5:
            print(",".join(desc + [str(minVal), str(minValIndex)]))

        # write to file
        outFile.write(",".join(desc + [str(minVal), str(minValIndex)]))
        outFile.write("\n")


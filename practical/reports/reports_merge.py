from pathlib import Path

csv_filenames = [
    Path("reports/report.csv"),
    Path("reports/report_dipl1.csv"),
    Path("reports/report_dipl2.csv"),
    Path("reports/report_dipl3.csv"),
    Path("reports/report_dipl4.csv"),
]

out_filename = Path("reports/report_merged.csv")

with open(out_filename, "w") as outFile:

    for csv_filename in csv_filenames:
        with open(csv_filename,  "r") as csvFile:
            for line in csvFile:
                outFile.write(line);
                

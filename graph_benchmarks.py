import os
import sys

from matplotlib import pyplot as plt

filenames = sys.argv[1:]


data = {}  # Test Name -> (File Name -> [(Numels, Time, CPU Time)])

for filename in filenames:
    with open(filename, "r") as f:
        lines = f.readlines()

    i = 0
    while "Iterations" not in lines[i]:
        i += 1

    i += 2

    while i < len(lines) and "ns" in lines[i]:
        line = lines[i]
        test_name = line[: line.index("/")]

        numel_start = line.index(":") + 1

        j = numel_start

        while line[j] != " " and line[j] != "/":
            j += 1

        numel = int(line[numel_start:j])

        j = line.index(" ", j)
        while line[j] == " ":
            j += 1
        time_start = j
        j = line.index(" ", time_start)

        time = int(line[time_start:j])

        j += 3  # skip "ns"
        while line[j] == " ":
            j += 1
        cpu_start = j
        j = line.index(" ", cpu_start)

        cpu = int(line[cpu_start:j])

        # Save to data

        if test_name not in data:
            data[test_name] = {}

        if filename not in data[test_name]:
            data[test_name][filename] = []

        data[test_name][filename].append((numel, time, cpu))

        i += 1

directory = "benchmark_results"

if not os.path.isdir(directory):
    os.mkdir(directory)

for (test_name, test_result) in data.items():
    for i in (0, 1):
        for test_data in test_result.values():
            plt.plot([td[0] for td in test_data], [td[i + 1] for td in test_data])
        plt.legend(test_result.keys())
        title = test_name + " - " + ["Time", "CPU Time"][i]
        plt.title(title)
        plt.savefig(directory + "/" + title + ".png")
        plt.clf()

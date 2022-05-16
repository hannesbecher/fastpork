"""
find_threshold.py: 
Find solid threshold from value statistics.
Output threshold and number of k-mers with count >= threshold.
Jens Zentgraf & Sven Rahmann, 2022
"""

from argparse import ArgumentParser
from collections import Counter


def get_argument_parser():
    p = ArgumentParser(description="find solid threshold")
    p.add_argument("logfile",
        help="logfile from counter with value statistics")
    p.add_argument("--start", "-s", type=int, default=1,
        help="first correct counter [1]")
    p.add_argument("--plot",
        help="file name of histogram plot file")
    return p


def plot(fname, L, start, end=100):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    Ls = L[start:end+1]
    ns = len(Ls)
    sns.barplot(x=list(range(start,start + ns)), y=Ls, color='blue')
    fig = plt.gcf()
    fig.set_size_inches(30,12)
    plt.yscale('log')
    plt.xlabel("$k$-mer count")
    plt.ylabel("number of $k$-mers")
    plt.title("$k$-mer Count Histogram")
    plt.savefig(fname, bbox_inches='tight')


def main(args):
    start = int(args.start)
    if start < 1:
        raise ValueError(f"--start argument must be an integer >= 1, but is {start=}")
    C = Counter()
    active = False
    with open(args.logfile, "rt") as flog:
        for line in flog:
            line = line.strip()
            if active:
                if not len(line):
                    active = False
                    continue
                fields = line.split()
                c = int(fields[0][:-1])
                f = int(fields[1])
                C[c] = f
                continue
            if line == "Value statistics:":
                active = True

    # make a list from counter
    assert C[0] == 0
    cs = sorted(C.keys())
    L = [0] * (cs[-1]+1)
    for c in cs:
        L[c] = C[c]
    assert L[0] == 0

    # find first minimum
    T = 1
    for c in range(start+1, cs[-1]+1):
        if L[c] > L[c-1]:
            T = c
            break
    S = sum(f for c,f in enumerate(L) if c>=T)
    print(T, S)

    if args.plot is not None:
        plot(args.plot, L, start)


if __name__ == "__main__":
    p = get_argument_parser()
    args = p.parse_args()
    main(args)

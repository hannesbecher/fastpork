# DESCRIPTION

The fastpork project provides a fast parallel pork k-mer counter.


# INSTALLATION

## Create conda environment

```bash
conda env create
conda activate fastpork
```

## Install scripts in developer mode

Inside the directory with our `setup.py` file, do
```bash
pip install -e .
```
(Note the dot at the end, for the current directory.)

# EXAMPLE

To count 27-mers on a large sequenced individual pig sample with files `mypig1.fq.gz` and `mypig2.fq.gz`, run something like this:
```bash
fastpork filtercount  -k 27 --filter <(zcat mypig1.fq.gz mypig2.fq.gz) --fastq <(zzcat mypig1.fq.gz mypig2.fq.gz) --filtersize1 21.0 --filtersize2 8.0  -v count 512 -n 4_400_000_000 -p 3  --maxwalk 2000 --fill 1.0 --subtables 19 -o mypig.counts.h5 > mypig.counts.log
```


# PERFORMANCE TIPS

## Decompressing gzipped FASTQ on-the-fly
For high speed, use anoymous pipes to unzip gzipped FASTQ files with `zcat` or `pigz -cd -p2` if your shell supports it.
The `pigz` tools gives slightly better performance because it uses 2 threads.
```bash
appname <(zcat *.fq.gz)
# or
appname <(pigz -cd -p2 *.fq.gz)
```

# paycheck
tAxonomY Classification Expected Composition

Code for [this study](https://doi.org/10.1101/406611). In support of [q2-clawback](https://library.qiime2.org/plugins/q2-clawback).

`paycheck` contains the cross-validation code.

`analysis` collates results.

`figures-and-tables` draws figures and generates tables.

Works with data found [here](https://doi.org/10.5281/zenodo.2548899) and [here](https://doi.org/10.5281/zenodo.2549777).

The main results in the [paper](https://doi.org/10.1101/406611) were obtained in the following fashion.

Install [QIIME 2](https://qiime2.org) and [q2-clawback](https://library.qiime2.org/plugins/q2-clawback). Install `paycheck` by running
```
pip install git+https://github.com/BenKaehler/paycheck.git
```
Download the Greengenes 13.8 reference data, which can also be found on the [QIIME 2 site](https://qiime2.org).
Make some working directories
```
mkdir empo-3
cd empo-3
mkdir ref
mkdir sediment-non-saline
```
Find the files `99_otus.fasta` and `99_otu_taxonomy.txt` amongst the Greengenes data and copy them into the reference directory (`ref`). Now import them into QIIME 2 Artifacts and extract V4 reads:
```
cd ref
qiime tools import \
  --type 'FeatureData[Sequence]' \
  --input-path 99_otus.fasta \
  --output-path 99_otus.qza

qiime feature-classifier extract-reads \
  --i-sequences 99_otus.qza \
  --p-f-primer GTGYCAGCMGCCGCGGTAA \
  --p-r-primer GGACTACNVGGGTWTCTAAT \
  --o-reads ref-seqs-v4.qza

qiime tools export \
  --input-path ref-seqs-v4.qza
  --output-path .

mv dna-sequences.fasta 99_otus_v4.fasta
```
Move into our working directory.
```
cd ../sediment-non-saline
```
Retrieve a dataset.
```
wget https://zenodo.org/record/2548899/files/sediment-non-saline.qza
```
An updated dataset for the same data could alteratively be obtained by running
```
qiime clawback fetch-Qiita-samples \
  --p-metadata-key empo_3 \
  --p-metadata-value 'Sediment (non-saline)' \
  --p-metadata-value 'sediment (non-saline)' \
  --o-samples sediment-non-saline.qza \
  --p-context Deblur-Illumina-16S-V4-150nt-780653
```
See the [q2-clawback tutorial](https://forum.qiime2.org/t/using-q2-clawback-to-assemble-taxonomic-weights/5859) for details.

Extract the sequence variant data.
```
qiime tools export \
  --input-path sediment-non-saline.qza \
  --output-path .
```
We can now run the cross-validation script.
```
mkdir results tmp
paycheck_cv \
  --empirical-samples feature-table.biom \
  --ref-seqs ../ref/99_otus_v4.fasta  \
  --ref-taxa ../ref/99_otu_taxonomy.txt \
  --results-dir results \
  --intermediate-dir tmp \
  --k 5 \
  --log-file results/log \
  --log-level DEBUG
```
If you have MPI installed on your system you can use `mpirun` and `--n-jobs` to accelerate the process. 

The cross validation results should now be in `bespoke70` and `uniform70` subdirectories of `results`.

The above steps can be repeated for any of the EMPO 3 types in the paper.

Next we analyse the results. Change directories to wherever you keep git repositories, clone the `paycheck` repo, and launch a Jupyter notebook server in the `analysis` subdirectory. Note that Jupyter is included in your QIIME 2 installation.
```
git clone https://github.com/BenKaehler/paycheck.git
cd paycheck/analysis
jupyter notebook
```
Now open the `eval-empo-3-error-rate.ipynb` notebook and change the values in the third cell to reflect he location of your `empo-3` directory, the EMPO 3 types for which you have run the analysis, and the weights that you have tried. So far we have only run `bespoke70` and `uniform70`, so truncate `class_weights` to only include those weights. If you have only run cross validation for Sediment (non-saline) you would truncate `sample_types` to only include `sediment-non-saline`.

Run the fourth cell to generate the `eval_taxa_er.tsv` result file.

The results can be visualised by running, for instance, the `error-rate-vs-habitat.ipynb` Jupyter notebook in the `figures-and-tables` subdirectory.

To calculate and visualise the other metrics presented in the paper, it is necessary to run the other `paycheck-eval-*` notebooks in the `analysis` subdirectory, followed by the required visualisation notebooks in the `figures-and-tables` subdirectory.

Finally, analysing the results for `average` taxonomic weights is somewhat more involved, because it requires that weights be drawn from all of the EMPO 3 habitats at once.

For this process, start by calculating the average weights for each EMPO 3 habitat. For example
```
cd empo-3/sediment-non-saline
qiime clawback sequence-variants-from-samples \
  --i-samples sediment-non-saline.qza \
  --o-sequences tmp/reads.qza
qiime feature-classifier classify-sklearn \
  --i-reads tmp/reads.qza \
  --i-classifier ../ref/uniform-classifier.qza \
  --p-confidence -1 \
  --o-classification tmp/classification.qza
qiime clawback generate-class-weights \
  --i-reference-taxonomy ../ref/99_otu_taxonomy.qza \
  --i-reference-sequences ../ref/99_otus_v4.qza \
  --i-samples sediment-non-saline.qza \
  --i-taxonomy-classification tmp/classification.qza \
  --o-class-weight results/weights.qza
```
The `uniform-classifier.qza` for the V4 region can be downloaded from the [QIIME 2](https://qiime2.org) website or trained im the standard way. See the q2-feature-classifier tutorial, which is available at the same site, for details.

Now, to calculate the average taxonomic weights resuls, start by telling the cross-validation script where to find the taxonomic weights for each EMPO3 habitat. This takes the form of a file that contains one file path on each line. For instance
```
cat > ../weights_files << END
../sediment-saline/results/weights.qza
END
```
Now, we can perform cross validation using average weihts by running
```
paycheck_cv_average \
  --ref-taxa ../ref/99_otu_taxonomy.txt \
  --ref-seqs ../ref/99_otus_v4.fasta \
  --weights ../weights_files \
  --exclude results/weights.qza \
  --obs-dir average \
  --results-dir results \
  --intermediate-dir tmp \
  --log-file results/log-average \
  --log-level DEBUG
```
This command will leave results in the `results/average` directory alongside `bespoke70` and `uniform70`. Note that `paycheck_cv` must be run before running `paycheck_cv_average` as it uses the same cross-validation partitions.

The average results can now be analysed in the same way that we analysed bespoke and uniform results, above.




import tempfile
import logging
import json
from os.path import join, exists

import click
import biom
from numpy import log
import numpy as np
import pandas as pd
from qiime2.plugins import diversity
from qiime2.plugins import phylogeny
import skbio

from .cross_validate import setup_logging


@click.command()
@click.option('--results-dir', required=True, type=click.Path(exists=True),
              help='Directory that will contain the result subdirectories')
@click.option('--intermediate-dir', default=tempfile.TemporaryDirectory(),
              type=click.Path(exists=True), help='Directory for checkpointing')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
              type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
              default='WARNING', help='Log level')
def shannon(results_dir, intermediate_dir, log_file, log_level):
    # set up logging
    setup_logging(log_level, log_file)
    logging.info(locals())

    # Reduce the empirical samples to pure taxonomy-level information
    biom_path = join(intermediate_dir, 'taxonomy_samples.biom')
    taxonomy_samples = biom.load_table(biom_path)
    logging.info('Got taxonomy samples')

    weighted_h, weighted_jsd = get_stats(taxonomy_samples)
    taxonomy_samples.norm()
    h, jsd = get_stats(taxonomy_samples)
    logging.info('Got stats')

    with open(join(results_dir, 'shannon.json'), 'w') as shannon_out:
        json.dump({'weighted_h': weighted_h, 'weighted_jsd': weighted_jsd,
                   'h': h, 'jsd': jsd, 'n': len(taxonomy_samples.ids())},
                  shannon_out)

    table = qiime2.Artifact.import_data(
        'FeatureTable[Frequency]', taxonomy_samples)

    beta_metrics = ['braycurtis', 'jaccard']
    beta_p_metrics = ['unweighted_unifrac', 'weighted_unifrac']
    alpha_metrics = ['shannon', 'observed_otus', 'pielou_e', 'gini_index']
    tree_path = join(intermediate_dir, '99_otus.tree')
    if exists(tree_path):
        beta_metrics.extend(beta_p_metrics)
        alpha_metrics.append('faith_pd')
        tree = qiime2.Artifact.import_data('Phylogeny[Unrooted]', tree_path)
        tree = qiime.phylogeny.methods.midpoint_root(tree=tree).rooted_tree

    distance_log = dict()
    for m in beta_metrics:
        if m in beta_p_metrics:
            dm = diversity.methods.beta_phylogenetic_alt(
                table=table, phylogeny = tree, metric=m)
        else:
            dm = diversity.methods.beta(table=table, metric=m)
        distances = dm.distance_matrix.view(
            skbio.DistanceMatrix).condensed_form()
        # we could do something like distance from centroid if we want to get
        # a vector of distances per sample. Otherwise quartiles might give us
        # something meaningful to look at.
        distance_log[m] = {'mean': distances.mean(),
                           'min': distances.min(),
                           'max': distances.max(),
                           '25percentile': np.percentile(distances, 25),
                           'median': np.percentile(distances, 50),
                           '75percentile': np.percentile(distances, 75),
                           'std': distances.std()}
    with open(join(results_dir, 'distances.json'), 'w') as distances_out:
        json.dump(distance_log, distances_out)

    alpha_vector = pd.DataFrame()
    for m in alpha_metrics:
        if m != 'faith_pd':
            alf = diversity.actions.alpha(table=table, metric=m)
        else:
            alf = diversity.actions.alpha_phylogenetic(
                table=table, phylogeny = tree, metric=m)
        alf = alf.alpha_diversity.view(pd.Series)
        alpha_vector = pd.concat([alpha_vector, alf.to_frame()])
    alpha_vector.to_csv(join(results_dir, 'alpha.tsv'), sep='\t')


def get_stats(taxonomy_samples):
    tax_weights = taxonomy_samples.sum(axis='observation')
    total = tax_weights.sum()
    tax_weights /= total
    h_avg = -(tax_weights*log(tax_weights)).sum()
    avg_h = 0.
    for column, _, _ in taxonomy_samples.iter():
        column = column[column != 0.]
        column_total = column.sum()
        p = column / column_total
        avg_h += -(p*log(p)).sum() * column_total/total
    return h_avg, h_avg - avg_h

import tempfile
import logging
import json
from os.path import join

import click
import biom
from numpy import log

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

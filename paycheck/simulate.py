import csv
from collections import defaultdict, Counter
import hashlib
import tempfile
import os
from os.path import join
import subprocess
import shutil
import logging
import socket
from traceback import format_exc
import sys

import click
import numpy.random
import numpy
import biom
import skbio.io
from pandas import DataFrame, Series
from qiime2 import Artifact
from qiime2.plugins import clawback
from mpiutils import dispatcher, mpi_logging


@click.command()
@click.option('--biom-file', required=True, type=click.Path(exists=True),
              help='Sample table with SVs for observation ids (biom)')
@click.option('--missed-sample-file', required=True,
              type=click.Path(exists=True),
              help='Basenames of the sample files that failed generation')
@click.option('--sv-to-ref-seq-file', required=True,
              type=click.Path(exists=True),
              help='BLAST mapping from SVs to ref seq labels (tsv)')
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
              help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
              help='Greengenes reference sequences (fasta)')
@click.option('--expected-dir', required=True, type=click.Path(exists=True),
              help='Output directory for expected taxa Artifacts')
@click.option('--abundances-dir', required=True, type=click.Path(exists=True),
              help='Output directory for expected taxa frequency Artifacts')
@click.option('--sequences-dir', required=True, type=click.Path(exists=True),
              help='Output directory for the simulated SV Artifacts')
@click.option('--tmp-dir', type=click.Path(exists=False),
              help='Temp dir (gets left behind on simulation exception)')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
              type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
              default='WARNING', help='Log level')
def simulate_missed_samples(biom_file, missed_sample_file, sv_to_ref_seq_file,
                            ref_taxa, ref_seqs, expected_dir, abundances_dir,
                            sequences_dir, tmp_dir=None, log_file=None,
                            log_level='DEBUG'):
    setup_logging(log_level, log_file)

    if dispatcher.am_dispatcher():
        logging.info(locals())
        all_samples = biom.load_table(biom_file)
        missed_samples = load_missed_samples(missed_sample_file)

    def process_sample(basename_sample):
        basename, sample = basename_sample
        try:
            exp_filename = join(expected_dir, basename)
            abund_filename = join(abundances_dir, basename)
            seqs_filename = join(sequences_dir, basename)
            generate_triple(
                basename[:-4], sample, sv_to_ref_seq_file, ref_taxa, ref_seqs,
                exp_filename, abund_filename, seqs_filename, tmp_dir)
            logging.info('Done ' + basename)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logging.warning('Skipping ' + basename + ':\n' + format_exc())

    def sample_generator():
        for fold, sample_id in missed_samples:
            basename = sample_id + '-fold-' + str(fold) + '.qza'
            yield basename, extract_sample([sample_id], all_samples)

    result = dispatcher.farm(process_sample, sample_generator())
    if result:
        list(result)


def load_missed_samples(missed_sample_file):
    with open(missed_sample_file) as ms_fh:
        missed_samples = []
        for line in ms_fh:
            line = line.strip()
            if line.endswith('.qza'):
                line = line[:-4]
            fold = int(line.rsplit('-', 1)[-1])
            sample_id = line.rsplit('-', 2)[0]
            missed_samples.append((fold, sample_id))
        return missed_samples


@click.command()
@click.option('--biom-file', required=True, type=click.Path(exists=True),
              help='Sample table with SVs for observation ids (biom)')
@click.option('--sv-to-ref-seq-file', required=True,
              type=click.Path(exists=True),
              help='BLAST mapping from SVs to ref seq labels (tsv)')
@click.option('--sv-to-ref-tax-file', required=True,
              type=click.Path(exists=True),
              help='Naive Bayes mapping from SVs to ref seq taxa (qza)')
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
              help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
              help='Greengenes reference sequences (fasta)')
@click.option('--weights-dir', required=True, type=click.Path(exists=True),
              help='Output directory for fold weights Artifacts')
@click.option('--expected-dir', required=True, type=click.Path(exists=True),
              help='Output directory for expected taxa Artifacts')
@click.option('--abundances-dir', required=True, type=click.Path(exists=True),
              help='Output directory for expected taxa frequency Artifacts')
@click.option('--sequences-dir', required=True, type=click.Path(exists=True),
              help='Output directory for the simulated SV Artifacts')
@click.option('--k', type=int, default=10,
              help='Number of folds for cross validation (default 10)')
@click.option('--tmp-dir', type=click.Path(exists=False),
              help='Temp dir (gets left behind on simulation exception)')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
              type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
              default='WARNING', help='Log level')
def simulate_all_samples(biom_file, sv_to_ref_seq_file, sv_to_ref_tax_file,
                         ref_taxa, ref_seqs, weights_dir,
                         expected_dir, abundances_dir, sequences_dir, k=10,
                         tmp_dir=None, log_file=None, log_level='DEBUG'):
    setup_logging(log_level, log_file)

    if dispatcher.am_dispatcher():
        logging.info(locals())
        all_samples = biom.load_table(biom_file)

        # shuffle the sample ids, assign folds, and generate weights
        sample_ids = numpy.array(all_samples.ids())
        logging.info('Found ' + str(len(sample_ids)) + ' samples')
        numpy.random.shuffle(sample_ids)
        folds = numpy.array([i % k for i in range(len(sample_ids))])
        reference_taxonomy = Artifact.import_data(
            'FeatureData[Taxonomy]', ref_taxa,
            view_type='HeaderlessTSVTaxonomyFormat')
        reference_sequences = Artifact.import_data(
            'FeatureData[Sequence]', ref_seqs)
        taxonomy_classification = Artifact.load(sv_to_ref_tax_file)
        for fold in range(k):
            training_set = extract_sample(
                sample_ids[folds != fold], all_samples)
            table = Artifact.import_data(
                'FeatureTable[Frequency]', training_set)
            unobserved_weight = 1e-6
            normalise = False
            weights = clawback.methods.generate_class_weights(
                reference_taxonomy, reference_sequences,
                table, taxonomy_classification)
            weights = weights.class_weight
            weights_filename = \
                'weights-normalise-%s-unobserved-weight-%g-fold-%d.qza' %\
                (normalise, unobserved_weight, fold)
            weights.save(join(weights_dir, weights_filename))

    def process_sample(basename_sample):
        basename, sample = basename_sample
        try:
            exp_filename = join(expected_dir, basename)
            abund_filename = join(abundances_dir, basename)
            seqs_filename = join(sequences_dir, basename)
            generate_triple(
                basename[:-4], sample, sv_to_ref_seq_file, ref_taxa, ref_seqs,
                exp_filename, abund_filename, seqs_filename, tmp_dir)
            logging.info('Done ' + basename)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            logging.warning('Skipping ' + basename + ':\n' + format_exc())

    def sample_generator():
        for fold, sample_id in zip(folds, sample_ids):
            basename = sample_id + '-fold-' + str(fold) + '.qza'
            yield basename, extract_sample([sample_id], all_samples)

    result = dispatcher.farm(process_sample, sample_generator())
    if result:
        list(result)


def setup_logging(log_level=None, log_file=None):
    try:
        if log_file:
            log_dir = os.path.dirname(log_file)
            dispatcher.checkmakedirs(log_dir)
            handler = mpi_logging.MPIFileHandler(log_file)
        else:
            handler = logging.StreamHandler()
        log_level = getattr(logging, log_level.upper())
        handler.setLevel(log_level)
        hostpid = ''
        if dispatcher.USING_MPI:
            hostpid = socket.gethostname()+':'+str(os.getpid())+':'
        formatter = logging.Formatter('%(asctime)s:' + hostpid +
                                      '%(levelname)s:%(message)s')
        handler.setFormatter(formatter)
        logging.root.addHandler(handler)
        logging.root.setLevel(log_level)
    except Exception:
        sys.stderr.write(' Unable to set up logging:\n'+format_exc())
        dispatcher.exit(1)


def extract_sample(sample_ids, samples):
    subsample = samples.filter(sample_ids, inplace=False)
    subsample.filter(
        lambda v, _, __: v.sum() > 1e-9, axis='observation', inplace=True)
    return subsample


def load_sv_map(sv_to_ref_seq_file):
    with open(sv_to_ref_seq_file) as blast_results:
        blast_reader = csv.reader(blast_results, csv.excel_tab)
        sv_map = {sv: ref_seq for sv, ref_seq in blast_reader}
    return sv_map


def load_ref_seqs_map(ref_seqs):
    with open(ref_seqs) as ref_fh:
        fasta_reader = skbio.io.read(ref_fh, 'fasta')
        ref_seqs = {s.metadata['id']: str(s) for s in fasta_reader}
    return ref_seqs


def generate_triple(sample_id, sample, sv_to_ref_seq_file, ref_taxa, ref_seqs,
                    exp_filename, abund_filename, seqs_filename, tmp_dir):
    sv_map = load_sv_map(sv_to_ref_seq_file)
    ref_seqs_map = load_ref_seqs_map(ref_seqs)
    tax_map = load_taxonomy_map(ref_taxa)

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            dada_in_dirs = simulate(sample, tmpdir, sv_map, ref_seqs_map)
            dada_out_dirs, dada_tmp_dirs = denoise(tmpdir, dada_in_dirs)
            result = traceback(dada_out_dirs, dada_tmp_dirs, tax_map)
            save_result(
                sample_id, result, exp_filename, abund_filename, seqs_filename)
        except Exception:
            if tmp_dir is not None and not os.path.exists(tmp_dir):
                shutil.copytree(tmpdir, tmp_dir)
            raise


def load_taxonomy_map(ref_taxa):
    with open(ref_taxa) as tax_fh:
        tax_map = {r[0].encode('utf-8'): r[1]
                   for r in csv.reader(tax_fh, csv.excel_tab)}
    return tax_map


def simulate(sample, tmpdir, sv_map, ref_seqs):
    'add some noise to the reference sequences'

    # output the amplicon sequences to fasta, labelled by greengenes sequence
    # label, with abundance that `vsearch --rereplicate` will understand
    abundance_filename = join(tmpdir, 'abundance.fasta')
    with open(abundance_filename, 'w') as a_fh:
        for row in sample.iter(axis='observation'):
            abundance, sv, _ = row
            abundance = int(abundance[0])
            if sv in sv_map:
                label = sv_map[sv]
                a_fh.write('>' + label + ';size=' + str(abundance) + '\n')
                a_fh.write(ref_seqs[sv_map[sv]] + '\n')

    # repreplicate according to abundance and run ART to simulate amplicons
    prior_art_filename = join(tmpdir, 'prior_art.fasta')
    cmd = ('vsearch', '--rereplicate', abundance_filename, '--output',
           prior_art_filename)
    subprocess.run(cmd, check=True)

    post_art_filename = join(tmpdir, 'post_art')
    cmd = ('art_illumina -ss MSv1 -amp -i ' + prior_art_filename +
           ' -l 250 -o ' + post_art_filename + ' -c 1 -na -p').split()
    subprocess.run(cmd, check=True)

    dada_in_dirs = []
    for i in ('1', '2'):
        cmd = 'gzip', post_art_filename + i + '.fq'
        subprocess.run(cmd, check=True)
        dada_in_dir = join(tmpdir, 'dada_in' + i)
        os.mkdir(dada_in_dir)
        dst = join(dada_in_dir, 'post_art' + i + '.fastq.gz')
        shutil.move(post_art_filename + i + '.fq.gz', dst)
        dada_in_dirs.append(dada_in_dir)

    return dada_in_dirs


def denoise(tmpdir, dada_in_dirs):
    'take the nose away, in a way that horribly mangles the sample provenance'
    post_dada_filename = join(tmpdir, 'post_dada.tsv')
    dada_tmp_dirs = [join(tmpdir, 'dada_tmp' + i) for i in ('1', '2')]
    dada_out_dirs = [join(tmpdir, 'dada_out' + i) for i in ('1', '2')]
    list(map(os.mkdir, dada_tmp_dirs + dada_out_dirs))

    cmd = 'run_traceable_dada_paired.R'.split() +\
        dada_in_dirs + [post_dada_filename] + dada_tmp_dirs +\
        '250 250 0 0 Inf 0 none 1 1 1000000'.split() + dada_out_dirs
    subprocess.run(cmd, check=True)

    return dada_out_dirs, dada_tmp_dirs


def traceback(dada_out_dirs, dada_tmp_dirs, tax_map):
    'reconstruct the taxa to which each denoised sequence corresponds'
    unique_maps = []
    for i, dada_out_dir in enumerate(dada_out_dirs, 1):
        with open(join(dada_out_dir, 'post_art%d.merge.map' % i)) as merg_fh:
            reader = csv.reader(merg_fh, csv.excel_tab)
            merg_map = {dada_sv: merg_sv for dada_sv, merg_sv in reader}
        with open(join(dada_out_dir, 'post_art%d.dada.map' % i)) as dada_fh:
            reader = csv.reader(dada_fh, csv.excel_tab)
            unique_map = defaultdict(list)
            for unique, dada_sv in reader:
                if dada_sv in merg_map:
                    unique_map[unique].append(merg_map[dada_sv])
        unique_maps.append(unique_map)

    # this is where the magic happens
    filtered_taxa = []
    single_maps = []
    for i, dada_tmp_dir in enumerate(dada_tmp_dirs, 1):
        single_map = defaultdict(set)
        taxa = []
        fastq_filename = join(dada_tmp_dir, 'post_art%d.fastq.gz' % i)
        with skbio.io.open(fastq_filename) as pa_fh:
            fastq_reader = skbio.io.read(pa_fh, 'fastq', phred_offset=33)
            for j, seq in enumerate(fastq_reader):
                taxa.append(tax_map[seq.metadata['id'][:-4].encode('utf-8')])
                for sv in unique_maps[i-1][str(seq)]:
                    single_map[sv].add(j)
        single_maps.append(single_map)
        filtered_taxa.append(taxa)
    assert filtered_taxa[0] == filtered_taxa[1]
    filtered_taxa = filtered_taxa[0]
    merged_map = {sv: single_maps[0][sv].intersection(single_maps[1][sv])
                  for sv in single_maps[0]}
    merged_map = {sv: Counter(filtered_taxa[i] for i in tlist)
                  for sv, tlist in merged_map.items()}
    result = [(s, t, c) for s in merged_map for t, c in merged_map[s].items()]
    return result


def save_result(
        sample_id, result, exp_filename, abund_filename, seqs_filename):
    'save the results in three Artifacts'
    svs, taxa, abundances = zip(*result)
    hashes = [hashlib.md5((s+t).encode('utf-8')).hexdigest()
              for s, t, c in result]

    expected = DataFrame({'Taxon': taxa}, index=hashes, columns=['Taxon'])
    expected.index.name = 'Feature ID'
    expected = Artifact.import_data('FeatureData[Taxonomy]', expected)
    expected.save(exp_filename)

    abundanced = DataFrame({h: a for h, a in zip(hashes, abundances)},
                           index=[sample_id], columns=hashes)
    abundanced = Artifact.import_data('FeatureTable[Frequency]', abundanced)
    abundanced.save(abund_filename)

    sequences = Series(svs, index=hashes)
    sequences = Artifact.import_data('FeatureData[Sequence]', sequences)
    sequences.save(seqs_filename)

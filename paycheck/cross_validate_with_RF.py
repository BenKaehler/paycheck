import glob
import csv
from collections import defaultdict, Counter
from itertools import cycle
import tempfile
import os
from os.path import join
import logging
from traceback import format_exc
import sys
import json

import click
import numpy.random
import numpy
import biom
from biom import Table
import skbio.io
from sklearn.model_selection import StratifiedKFold, KFold
from pandas import DataFrame
from qiime2 import Artifact
from qiime2.plugins import clawback, feature_classifier
from q2_types.feature_data import DNAIterator
from scipy.sparse import dok_matrix


@click.command()
@click.option('--empirical-samples', required=True,
			  type=click.Path(exists=True),
			  help='Sample table with SVs for observation ids (biom)')
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
			  help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
			  help='Greengenes reference sequences (fasta)')
@click.option('--results-dir', required=True, type=click.Path(exists=True),
			  help='Directory that will contain the result subdirectories')
@click.option('--intermediate-dir', default=tempfile.TemporaryDirectory(),
			  type=click.Path(exists=True), help='Directory for checkpointing')
@click.option('--k', type=int, default=5,
			  help='Number of folds for cross validation (default 5)')
@click.option('--n-jobs', type=int, default=1,
			  help='Number of jobs for parallel classification')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
			  type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
			  default='WARNING', help='Log level')

def generate_folds(empirical_samples, ref_taxa, ref_seqs, results_dir,
				   intermediate_dir, k, n_jobs, log_file, log_level):

	# set up logging
	setup_logging(log_level, log_file)
	logging.info(locals())


	# Reduce the empirical samples to pure taxonomy-level information
	biom_path = join(intermediate_dir, 'taxonomy_samples.biom')
	if not os.path.exists(biom_path):
		taxonomy_samples = map_svs_to_taxa(
			empirical_samples, ref_taxa, ref_seqs, n_jobs)
		with biom.util.biom_open(biom_path, 'w') as biom_file:
			taxonomy_samples.to_hdf5(biom_file, 'paycheck')
	taxonomy_samples = biom.load_table(biom_path)
	logging.info('Got taxonomy samples')


	# Generate folds

	# Generate reference sequence folds
	seq_ids, seq_strata, taxon_defaults = \
		get_sequence_strata(k, ref_taxa, ref_seqs, n_jobs)
	skf = StratifiedKFold(n_splits=k, shuffle=True)
	seq_split = skf.split(seq_ids, seq_strata)

	# Generate empirical sample folds
	kf = KFold(n_splits=k, shuffle=True)
	sample_ids = taxonomy_samples.ids()
	sample_split = kf.split(sample_ids)


	# Save them down
	for i, ((seq_train, seq_test), (sample_train, sample_test)) in \
			enumerate(zip(seq_split, sample_split)):
		fold = join(intermediate_dir, 'fold-' + str(i))
		os.mkdir(fold)
		with open(join(fold, 'seq_train.json'), 'w') as fh:
			json.dump([seq_ids[j] for j in seq_train], fh)
		with open(join(fold, 'seq_test.json'), 'w') as fh:
			json.dump([seq_ids[j] for j in seq_test], fh)
		with open(join(fold, 'sample_train.json'), 'w') as fh:
			json.dump([sample_ids[j] for j in sample_train], fh)
		with open(join(fold, 'sample_test.json'), 'w') as fh:
			json.dump([sample_ids[j] for j in sample_test], fh)

	taxon_defaults_file = join(intermediate_dir, 'taxon_defaults.json')

	with open(taxon_defaults_file, 'w') as fh:
		json.dump(taxon_defaults, fh)

	folds = glob.glob(join(intermediate_dir, 'fold-*'))
	logging.info('Got folds')


	# For each fold
	for fold in folds:
		# Simulate the test samples
		test_samples, expected = simulate_samples(
			taxonomy_samples, fold, taxon_defaults, ref_taxa, ref_seqs)

		# Generate the class weights from the training samples
		train_taxa, train_seqs, ref_seqs_art, weights = get_train_artifacts(
			taxonomy_samples, fold, taxon_defaults, ref_taxa, ref_seqs)

		# Save out the expected taxonomies and abundances
		save_expected(results_dir, test_samples, expected, train_taxa)
		logging.info('Done expected results for ' + fold)

		# Save the test seqs, training taxa, training seqs, and weights
		weights_file = join(fold,'weights.qza')
		weights.save(weights_file)
		#training_seqs_file = join(fold,'train_seqs.qza')
		#train_seqs.save(training_seqs_file)
		training_taxa_file = join(fold,'train_taxa.qza')
		train_taxa.save(training_taxa_file)
		#test_seqs_file = join(fold,'ref_seqs_art.qza')
		#ref_seqs_art.save(test_seqs_file)
		logging.info('Done '+ fold)


@click.command()
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
			  help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
			  help='Greengenes reference sequences (fasta)')
@click.option('--classifier-spec', required=True, type=click.File('r'),
			  help='JSON-formatted q2-feature-classifier classifier spec')
@click.option('--obs-dir', required=True, type=str,
			  help='Subdirectory into which the results will be saved')
@click.option('--results-dir', required=True, type=click.Path(exists=True),
			  help='Directory that will contain the result subdirectories')
@click.option('--intermediate-dir', default=tempfile.TemporaryDirectory(),
			  type=click.Path(exists=True), help='Directory for checkpointing')
@click.option('--n-jobs', type=int, default=1,
			  help='Number of jobs for parallel classification')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
			  type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
			  default='WARNING', help='Log level')

@click.option('--confidence', required=True, type=float, help='Number of Confidence')
@click.option('--classifier-directory', required=True, type=str, help='Directory of Classifier')

def cross_validate_classifier(
		ref_taxa, ref_seqs, classifier_spec, obs_dir, results_dir,
		intermediate_dir, n_jobs, log_file, log_level, confidence, classifier_directory):

	classifier_spec = classifier_spec.read()

	# set up logging
	setup_logging(log_level, log_file)
	logging.info(locals())


	# load folds
	taxon_defaults_file = join(intermediate_dir, 'taxon_defaults.json')
	with open(taxon_defaults_file) as fh:
		taxon_defaults = json.load(fh)
	folds = glob.glob(join(intermediate_dir, 'fold-*'))
	logging.info('Got folds')

	# load ref_seq
	_ , ref_seqs = load_references(ref_taxa, ref_seqs)
	ref_seqs = Artifact.import_data(
		'FeatureData[Sequence]', DNAIterator(ref_seqs))

    
	# for each fold
	for fold in folds:
		# load new file for different folds
		weights_file = join(fold,'weights.qza')
		training_taxa_file = join(fold,'train_taxa.qza')

		# load the simulated test samples
		test_samples = load_simulated_samples(fold, results_dir)

		# load the test seqs, training taxa, traing seqs, and weights
		weights = Artifact.load(weights_file)
		#test_seqs = Artifact.load(test_seqs_file)
		train_taxa = Artifact.load(training_taxa_file)

		# train the weighted classifier and classify the test samples
		classification = classify_samples_sklearn(
			test_samples, train_taxa, ref_seqs, classifier_spec, confidence, n_jobs, weights)
		# save the classified taxonomy artifacts
		save_observed(classifier_directory, test_samples, classification, obs_dir)
		logging.info('Done ' + fold)



@click.command()
@click.option('--empirical-samples', required=True,
			  type=click.Path(exists=True),
			  help='Sample table with SVs for observation ids (biom)')
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
			  help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
			  help='Greengenes reference sequences (fasta)')
@click.option('--results-dir', required=True, type=click.Path(exists=True),
			  help='Directory that will contain the result subdirectories')
@click.option('--intermediate-dir', default=tempfile.TemporaryDirectory(),
			  type=click.Path(exists=True), help='Directory for checkpointing')
@click.option('--k', type=int, default=5,
			  help='Number of folds for cross validation (default 5)')
@click.option('--n-jobs', type=int, default=1,
			  help='Number of jobs for parallel classification')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
			  type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
			  default='WARNING', help='Log level')

def cross_validate(empirical_samples, ref_taxa, ref_seqs, results_dir,
				   intermediate_dir, k, n_jobs, log_file, log_level):
	# set up logging
	setup_logging(log_level, log_file)
	logging.info(locals())

	# Reduce the empirical samples to pure taxonomy-level information
	biom_path = join(intermediate_dir, 'taxonomy_samples.biom')
	if not os.path.exists(biom_path):
		taxonomy_samples = map_svs_to_taxa(
			empirical_samples, ref_taxa, ref_seqs, n_jobs)
		with biom.util.biom_open(biom_path, 'w') as biom_file:
			taxonomy_samples.to_hdf5(biom_file, 'paycheck')
	taxonomy_samples = biom.load_table(biom_path)
	logging.info('Got taxonomy samples')

	# Generate folds
	taxon_defaults_file = join(intermediate_dir, 'taxon_defaults.json')
	if not os.path.exists(taxon_defaults_file):
		# Generate reference sequence folds
		seq_ids, seq_strata, taxon_defaults = \
			get_sequence_strata(k, ref_taxa, ref_seqs, n_jobs)
		skf = StratifiedKFold(n_splits=k, shuffle=True)
		seq_split = skf.split(seq_ids, seq_strata)

		# Generate empirical sample folds
		kf = KFold(n_splits=k, shuffle=True)
		sample_ids = taxonomy_samples.ids()
		sample_split = kf.split(sample_ids)

		# Save them down
		for i, ((seq_train, seq_test), (sample_train, sample_test)) in \
				enumerate(zip(seq_split, sample_split)):
			fold = join(intermediate_dir, 'fold-' + str(i))
			os.mkdir(fold)
			with open(join(fold, 'seq_train.json'), 'w') as fh:
				json.dump([seq_ids[j] for j in seq_train], fh)
			with open(join(fold, 'seq_test.json'), 'w') as fh:
				json.dump([seq_ids[j] for j in seq_test], fh)
			with open(join(fold, 'sample_train.json'), 'w') as fh:
				json.dump([sample_ids[j] for j in sample_train], fh)
			with open(join(fold, 'sample_test.json'), 'w') as fh:
				json.dump([sample_ids[j] for j in sample_test], fh)
		with open(taxon_defaults_file, 'w') as fh:
			json.dump(taxon_defaults, fh)
	with open(taxon_defaults_file) as fh:
		taxon_defaults = json.load(fh)
	folds = glob.glob(join(intermediate_dir, 'fold-*'))
	logging.info('Got folds')

	# For each fold
	for fold in folds:
		# Simulate the test samples
		test_samples, expected = simulate_samples(
			taxonomy_samples, fold, taxon_defaults, ref_taxa, ref_seqs)
		# Generate the class weights from the training samples
		train_taxa, train_seqs, ref_seqs_art, weights = get_train_artifacts(
			taxonomy_samples, fold, taxon_defaults, ref_taxa, ref_seqs)
		# Save out the expected taxonomies and abundances
		save_expected(results_dir, test_samples, expected, train_taxa)
		logging.info('Done expected results for ' + fold)
		# Train the uniform classifier and classify the test samples
		classification = classify_samples(
			test_samples, train_taxa, ref_seqs_art, 0.7, n_jobs)
		# Save out the classified taxonomy artifacts
		save_observed(results_dir, test_samples, classification, 'uniform70')
		# Train the bespoke classifier and classify the test samples
		classification = classify_samples(
			test_samples, train_taxa, ref_seqs_art, 0.7, n_jobs, weights)
		# Save out the classified taxonomy artifacts
		save_observed(results_dir, test_samples, classification, 'bespoke70')
		# Do it all again
		classification = classify_samples(
			test_samples, train_taxa, ref_seqs_art, 0.92, n_jobs)
		# Save out the classified taxonomy artifacts
		save_observed(results_dir, test_samples, classification, 'uniform92')
		# Train the bespoke classifier and classify the test samples
		classification = classify_samples(
			test_samples, train_taxa, ref_seqs_art, 0.92, n_jobs, weights)
		# Save out the classified taxonomy artifacts
		save_observed(results_dir, test_samples, classification, 'bespoke92')
		logging.info('Done ' + fold)


@click.command()
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
			  help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
			  help='Greengenes reference sequences (fasta)')
@click.option('--weights', required=True, type=click.Path(exists=True),
			  help='Taxonomy weights for classification (qza)')
@click.option('--obs-dir', required=True, type=str,
			  help='Subdirectory into which the results will be saved')
@click.option('--results-dir', required=True, type=click.Path(exists=True),
			  help='Directory that will contain the result subdirectories')
@click.option('--intermediate-dir', default=tempfile.TemporaryDirectory(),
			  type=click.Path(exists=True), help='Directory for checkpointing')
@click.option('--n-jobs', type=int, default=1,
			  help='Number of jobs for parallel classification')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
			  type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
			  default='WARNING', help='Log level')

def cross_validate_for_weights(
		ref_taxa, ref_seqs, weights, obs_dir, results_dir,
		intermediate_dir, n_jobs, log_file, log_level):
	# set up logging
	setup_logging(log_level, log_file)
	logging.info(locals())

	# load taxonomy-level information
	biom_path = join(intermediate_dir, 'taxonomy_samples.biom')
	taxonomy_samples = biom.load_table(biom_path)
	logging.info('Got taxonomy samples')

	# load folds
	taxon_defaults_file = join(intermediate_dir, 'taxon_defaults.json')
	with open(taxon_defaults_file) as fh:
		taxon_defaults = json.load(fh)
	folds = glob.glob(join(intermediate_dir, 'fold-*'))
	logging.info('Got folds')

	# load the weights
	weights = Artifact.load(weights)
	# for each fold
	for fold in folds:
		# load the simulated test samples
		test_samples = load_simulated_samples(fold, results_dir)
		# generate the training taxa, seqs, ref_seqs, reduced weights
		train_taxa, train_seqs, ref_seqs_art, fold_weights = \
			get_train_artifacts(taxonomy_samples, fold, taxon_defaults,
								ref_taxa, ref_seqs, weights)
		# train the weighted classifier and classify the test samples
		classification = classify_samples(
			test_samples, train_taxa, ref_seqs_art, 0.7, n_jobs, fold_weights)
		# save the classified taxonomy artifacts
		save_observed(results_dir, test_samples, classification, obs_dir)
		logging.info('Done ' + fold)


@click.command()
@click.option('--ref-taxa', required=True, type=click.Path(exists=True),
			  help='Greengenes reference taxa (tsv)')
@click.option('--ref-seqs', required=True, type=click.Path(exists=True),
			  help='Greengenes reference sequences (fasta)')
@click.option('--weights', required=True, type=click.Path(exists=True),
			  help='Weights files (list of qzas, one per line)')
@click.option('--exclude', required=True, type=click.Path(exists=True),
			  help='Weights file to exclude from average (qza)')
@click.option('--obs-dir', required=True, type=str,
			  help='Subdirectory into which the results will be saved')
@click.option('--results-dir', required=True, type=click.Path(exists=True),
			  help='Directory that will contain the result subdirectories')
@click.option('--intermediate-dir', default=tempfile.TemporaryDirectory(),
			  type=click.Path(exists=True), help='Directory for checkpointing')
@click.option('--n-jobs', type=int, default=1,
			  help='Number of jobs for parallel classification')
@click.option('--log-file', type=click.Path(), help='Log file')
@click.option('--log-level',
			  type=click.Choice('DEBUG INFO WARNING ERROR CRITICAL'.split()),
			  default='WARNING', help='Log level')

def cross_validate_average(
		ref_taxa, ref_seqs, weights, exclude, obs_dir, results_dir,
		intermediate_dir, n_jobs, log_file, log_level):
	# set up logging
	setup_logging(log_level, log_file)
	logging.info(locals())

	# load taxonomy-level information
	biom_path = join(intermediate_dir, 'taxonomy_samples.biom')
	taxonomy_samples = biom.load_table(biom_path)
	logging.info('Got taxonomy samples')

	# load folds
	taxon_defaults_file = join(intermediate_dir, 'taxon_defaults.json')
	with open(taxon_defaults_file) as fh:
		taxon_defaults = json.load(fh)
	folds = glob.glob(join(intermediate_dir, 'fold-*'))
	logging.info('Got folds')

	# load the weights
	other_weights = None
	with open(weights) as fh:
		for i, weights_file in enumerate(fh):
			weights_file = weights_file.rstrip()
			if os.path.samefile(weights_file, exclude):
				continue
			if other_weights is None:
				other_weights = Artifact.load(weights_file).view(Table)
				other_weights.update_ids(
					{'Weight': str(i)}, axis='sample', inplace=True)
			else:
				these_weights = Artifact.load(weights_file).view(Table)
				these_weights.update_ids(
					{'Weight': str(i)}, axis='sample', inplace=True)
				other_weights = other_weights.merge(these_weights)
	logging.info('Loaded ' + str(other_weights.shape[1]) + ' other weights')

	# for each fold
	for fold in folds:
		# load the simulated test samples
		test_samples = load_simulated_samples(fold, results_dir)
		# generate the class weights from the training samples
		train_taxa, train_seqs, ref_seqs_art, fold_weights = \
			get_train_artifacts(taxonomy_samples, fold, taxon_defaults,
								ref_taxa, ref_seqs)
		# join fold weights with other weights
		fold_weights = fold_weights.view(Table)
		fold_weights.update_ids(
			{'Weight': 'fold_weights'}, axis='sample', inplace=True)
		fold_weights = fold_weights.merge(other_weights)
		fold_weights = Artifact.import_data(
			'FeatureTable[RelativeFrequency]', fold_weights)
		# generate the training taxa, seqs, ref_seqs, reduced weights
		train_taxa, train_seqs, ref_seqs_art, fold_weights = \
			get_train_artifacts(taxonomy_samples, fold, taxon_defaults,
								ref_taxa, ref_seqs, fold_weights)
		# train the weighted classifier and classify the test samples
		classification = classify_samples(
			test_samples, train_taxa, ref_seqs_art, 0.7, n_jobs, fold_weights)
		# save the classified taxonomy artifacts
		save_observed(results_dir, test_samples, classification, obs_dir)
		logging.info('Done ' + fold)

def save_observed(results_dir, test_samples, classification, dirname):
	classification = classification.view(DataFrame)
	tax_map = classification['Taxon']

	observed_dir = join(results_dir, dirname)
	if not os.path.exists(observed_dir):
		os.mkdir(observed_dir)
	for sample_id in test_samples.ids():
		sample = extract_sample([sample_id], test_samples)
		ids = sample.ids(axis='observation')
		taxa = [tax_map[s] for s in ids]
		df = DataFrame({'Taxon': taxa}, index=ids, columns=['Taxon'])
		df.index.name = 'Feature ID'
		Artifact.import_data('FeatureData[Taxonomy]', df).save(
			join(observed_dir, sample_id + '.qza'))


def classify_samples(test_samples, train_taxa, ref_seqs, confidence,
					 n_jobs, weights=None):
	classifier = feature_classifier.methods.fit_classifier_naive_bayes(
		ref_seqs, train_taxa, class_weight=weights,
		classify__alpha=0.001, feat_ext__ngram_range='[7,7]')
	classifier = classifier.classifier

	test_ids = set(test_samples.ids(axis='observation'))
	test_seqs = ref_seqs.view(DNAIterator)
	test_seqs = (s for s in test_seqs if s.metadata['id'] in test_ids)
	test_seqs = DNAIterator(test_seqs)
	test_seqs = Artifact.import_data('FeatureData[Sequence]', test_seqs)

	logging.info('Commencing classification of ' +
				 str(len(list(test_seqs.view(DNAIterator)))) + ' sequences')
	classification = feature_classifier.methods.classify_sklearn(
		test_seqs, classifier, confidence=confidence, n_jobs=n_jobs)
	classification = classification.classification
	logging.info('Got some classifications')
	return classification


def classify_samples_sklearn(test_samples, train_taxa, ref_seqs, classifier_spec, confidence, n_jobs, weights=None):
	classifier = feature_classifier.methods.fit_classifier_sklearn(
		ref_seqs, train_taxa, class_weight=weights,
		classifier_specification=classifier_spec)
	classifier = classifier.classifier

	test_ids = set(test_samples.ids(axis='observation'))
	test_seqs = ref_seqs.view(DNAIterator)
	test_seqs = (s for s in test_seqs if s.metadata['id'] in test_ids)
	test_seqs = DNAIterator(test_seqs)
	test_seqs = Artifact.import_data('FeatureData[Sequence]', test_seqs)

	logging.info('Commencing classification of ' +
				 str(len(list(test_seqs.view(DNAIterator)))) + ' sequences')
	classification = feature_classifier.methods.classify_sklearn(
		test_seqs, classifier, confidence=confidence, n_jobs=n_jobs)
	classification = classification.classification
	logging.info('Got some classifications')
	return classification


def save_expected(results_dir, test_samples, expected, train_taxa):
	known_taxa = set()
	for taxon in set(train_taxa.view(DataFrame)['Taxon'].values):
		while ';' in taxon:
			known_taxa.add(taxon)
			taxon, _ = taxon.rsplit(';', 1)
		known_taxa.add(taxon)

	for sid in expected:
		taxon = expected[sid]
		while taxon not in known_taxa:
			taxon, _ = taxon.rsplit(';', 1)
		expected[sid] = taxon

	expected_dir = join(results_dir, 'expected')
	if not os.path.exists(expected_dir):
		os.mkdir(expected_dir)
	abundance_dir = join(results_dir, 'abundance')
	if not os.path.exists(abundance_dir):
		os.mkdir(abundance_dir)
	for sample_id in test_samples.ids():
		sample = extract_sample([sample_id], test_samples)
		ids = sample.ids(axis='observation')
		taxa = [expected[s] for s in ids]
		df = DataFrame({'Taxon': taxa}, index=ids, columns=['Taxon'])
		df.index.name = 'Feature ID'
		Artifact.import_data('FeatureData[Taxonomy]', df).save(
			join(expected_dir, sample_id + '.qza'))
		df = DataFrame(dict(zip(ids, sample.data(sample_id))),
					   index=['Frequency'], columns=ids)
		Artifact.import_data('FeatureTable[Frequency]', df).save(
			join(abundance_dir, sample_id + '.qza'))


def load_simulated_samples(fold, results_dir):
	with open(join(fold, 'sample_test.json')) as fp:
		sample_ids = json.load(fp)
	expected_dir = join(results_dir, 'expected')
	test_samples = defaultdict(set)
	for sample_id in sample_ids:
		expected = Artifact.load(join(expected_dir, sample_id + '.qza'))
		for obs_id in expected.view(DataFrame).index:
			test_samples[obs_id].add(sample_id)
	obs_ids = list(test_samples)
	data = dok_matrix((len(obs_ids), len(sample_ids)))
	s_map = {s: i for i, s in enumerate(sample_ids)}
	o_map = {o: i for i, o in enumerate(obs_ids)}
	for obs_id in test_samples:
		for sample_id in test_samples[obs_id]:
			data[o_map[obs_id], s_map[sample_id]] = 1
	test_samples = Table(data, obs_ids, sample_ids)
	return test_samples


def simulate_samples(
		taxonomy_samples, fold, taxon_defaults, ref_taxa, ref_seqs):
	with open(join(fold, 'sample_test.json')) as fp:
		test_samples = json.load(fp)
	test_samples = extract_sample(test_samples, taxonomy_samples)
	ref_taxa, _ = load_references(ref_taxa, ref_seqs)

	with open(join(fold, 'seq_test.json')) as fp:
		test_seqs = json.load(fp)
	test_taxa = {ref_taxa[sid] for sid in test_seqs}

	hits = [0]
	direct_remaps = [0]
	indirect_remaps = [0]

	def collapse(taxon, _):
		if taxon in test_taxa:
			hits[0] += 1
			return taxon
		if taxon_defaults[taxon][0] in test_taxa:
			direct_remaps[0] += 1
			return taxon_defaults[taxon][0]
		for try_taxon in taxon_defaults[taxon][1:]:
			if try_taxon in test_taxa:
				indirect_remaps[0] += 1
				return try_taxon
	test_samples = test_samples.collapse(
		collapse, norm=False, axis='observation')
	logging.info('Test taxon remaps')
	logging.info(str(hits[0]) + ' hits')
	logging.info(str(direct_remaps[0]) + ' direct remaps')
	logging.info(str(indirect_remaps[0]) + ' indirect remaps')

	samples = []
	obs_ids = []
	expected = []
	taxa_ref = defaultdict(list)
	for sid, taxon in ref_taxa.items():
		if sid in test_seqs:
			taxa_ref[taxon].append(sid)
	for abundances, taxon, _ in test_samples.iter(axis='observation'):
		taxa = taxa_ref[taxon]
		n_taxa = len(taxa)
		obs_ids.extend(taxa)
		expected.extend(ref_taxa[sid] for sid in taxa)
		taxa_samples = numpy.vstack([abundances // n_taxa]*n_taxa)
		# magic
		taxa = cycle(range(n_taxa))
		for i, r in enumerate(abundances % n_taxa):
			for t, _ in zip(taxa, range(int(r))):
				taxa_samples[t, i] += 1
		assert (taxa_samples.sum(axis=0) == abundances).all()
		samples.append(taxa_samples)
	test_samples = Table(numpy.vstack(samples), obs_ids, test_samples.ids())
	test_samples.filter(
		lambda v, _, __: v.sum() > 1e-9, axis='observation', inplace=True)

	return (test_samples, dict(zip(obs_ids, expected)))


def get_train_artifacts(taxonomy_samples, fold, taxon_defaults, ref_taxa,
						ref_seqs, weights=None):

	if weights is None:
		with open(join(fold, 'sample_train.json')) as fp:
			train_samples = json.load(fp)
		train_samples = extract_sample(train_samples, taxonomy_samples)
	else:
		train_samples = weights.view(Table)
	ref_taxa, ref_seqs = load_references(ref_taxa, ref_seqs)

	with open(join(fold, 'seq_train.json')) as fp:
		train_seqs = json.load(fp)
	train_taxa = {ref_taxa[sid] for sid in train_seqs}

	hits = [0]
	direct_remaps = [0]
	indirect_remaps = [0]

	def collapse(taxon, _):
		if taxon in train_taxa:
			hits[0] += 1
			return taxon
		if taxon_defaults[taxon][0] in train_taxa:
			direct_remaps[0] += 1
			return taxon_defaults[taxon][0]
		for try_taxon in taxon_defaults[taxon][1:]:
			if try_taxon in train_taxa:
				indirect_remaps[0] += 1
				return try_taxon
	train_samples = train_samples.collapse(
		collapse, axis='observation', norm=False)
	logging.info('Train taxon remaps')
	logging.info(str(hits[0]) + ' hits')
	logging.info(str(direct_remaps[0]) + ' direct remaps')
	logging.info(str(indirect_remaps[0]) + ' indirect remaps')
	train_samples = Artifact.import_data(
		'FeatureTable[Frequency]', train_samples)

	train_taxa = list(train_taxa)
	eye_taxonomy = DataFrame(
		{'Taxon': train_taxa}, index=train_taxa, columns=['Taxon'])
	eye_taxonomy.index.name = 'Feature ID'
	eye_taxonomy = Artifact.import_data('FeatureData[Taxonomy]', eye_taxonomy)
	train_taxa = [ref_taxa[sid] for sid in train_seqs]
	train_taxonomy = DataFrame(
		{'Taxon': train_taxa}, index=train_seqs, columns=['Taxon'])
	train_taxonomy.index.name = 'Feature ID'
	train_taxonomy = Artifact.import_data(
		'FeatureData[Taxonomy]', train_taxonomy)
	train_iter = DNAIterator(
		s for s in ref_seqs if s.metadata['id'] in train_seqs)
	train_art = Artifact.import_data(
		'FeatureData[Sequence]', train_iter)
	unobserved_weight = 1e-6 if weights is None else 0.
	weights = clawback.methods.generate_class_weights(
		train_taxonomy, train_art, train_samples, eye_taxonomy,
		unobserved_weight=unobserved_weight)
	ref_seqs = Artifact.import_data(
		'FeatureData[Sequence]', DNAIterator(ref_seqs))
	return train_taxonomy, train_art, ref_seqs, weights.class_weight


def map_svs_to_taxa(empirical_samples, ref_taxa, ref_seqs, n_jobs):
	ref_taxa = Artifact.import_data('FeatureData[Taxonomy]', ref_taxa,
									view_type='HeaderlessTSVTaxonomyFormat')
	ref_seqs = Artifact.import_data('FeatureData[Sequence]', ref_seqs)
	classifier = feature_classifier.methods.fit_classifier_naive_bayes(
		ref_seqs, ref_taxa,
		classify__alpha=0.001, feat_ext__ngram_range='[7,7]')
	classifier = classifier.classifier
	samples = Artifact.import_data(
		'FeatureTable[Frequency]', empirical_samples)
	svs = clawback.methods.sequence_variants_from_samples(samples)
	svs = svs.sequences
	sv_taxa = feature_classifier.methods.classify_sklearn(
		svs, classifier, confidence=0., n_jobs=n_jobs).classification
	sv_taxa = sv_taxa.view(DataFrame)['Taxon']
	return samples.view(Table).collapse(
		lambda sid, _: sv_taxa[sid], axis='observation', norm=False)


def get_sequence_strata(k, ref_taxa, ref_seqs, n_jobs):
	taxonomy, ref_seqs = load_references(ref_taxa, ref_seqs)
	taxa_stats = Counter(taxonomy.values())
	strata = {t: [t] for t in taxonomy.values() if taxa_stats[t] >= k}
	kref = (s for s in ref_seqs if taxonomy[s.metadata['id']] in strata)
	ref_art = Artifact.import_data('FeatureData[Sequence]', DNAIterator(kref))
	tax_art = Artifact.import_data('FeatureData[Taxonomy]', ref_taxa,
								   view_type='HeaderlessTSVTaxonomyFormat')
	classifier = feature_classifier.methods.fit_classifier_naive_bayes(
		ref_art, tax_art, classify__alpha=0.001, feat_ext__ngram_range='[7,7]')
	classifier = classifier.classifier
	tiddlers = DNAIterator(s for s in ref_seqs
						   if taxonomy[s.metadata['id']] not in strata)
	tid_art = Artifact.import_data('FeatureData[Sequence]', tiddlers)
	tid_tax = feature_classifier.methods.classify_sklearn(
		tid_art, classifier, confidence=0., n_jobs=n_jobs)
	tid_tax = tid_tax.classification.view(DataFrame)
	stratum_votes = defaultdict(Counter)
	for sid in tid_tax.index:
		stratum_votes[taxonomy[sid]][tid_tax['Taxon'][sid]] += \
			float(tid_tax['Confidence'][sid])
	taxon_defaults = {}
	for taxon in stratum_votes:
		most_common = stratum_votes[taxon].most_common()
		merge_taxon, max_conf = most_common[0]
		assert len(most_common) == 1 or most_common[1][1] != max_conf
		taxon_defaults[taxon] = strata[merge_taxon]
		strata[merge_taxon].append(taxon)
	taxon_defaults.update(strata)
	seq_ids = [s.metadata['id'] for s in ref_seqs]
	strata = [taxon_defaults[taxonomy[sid]][0] for sid in seq_ids]
	return seq_ids, strata, taxon_defaults


def load_references(ref_taxa, ref_seqs):
	with open(ref_seqs) as ref_fh:
		ref_seqs = list(skbio.io.read(ref_fh, 'fasta'))
	ref_ids = {s.metadata['id'] for s in ref_seqs}

	with open(ref_taxa) as tax_fh:
		tax_reader = csv.reader(tax_fh, csv.excel_tab)
		ref_taxa = {s: t for s, t in tax_reader if s in ref_ids}
	ref_seqs = [s for s in ref_seqs if s.metadata['id'] in ref_taxa]

	return ref_taxa, ref_seqs


def setup_logging(log_level=None, log_file=None):
	try:
		if log_file:
			handler = logging.FileHandler(log_file)
		else:
			handler = logging.StreamHandler()
		log_level = getattr(logging, log_level.upper())
		handler.setLevel(log_level)
		formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
		handler.setFormatter(formatter)
		logging.root.addHandler(handler)
		logging.root.setLevel(log_level)
	except Exception:
		sys.stderr.write(' Unable to set up logging:\n'+format_exc())
		sys.exit(1)


def extract_sample(sample_ids, samples):
	subsample = samples.filter(sample_ids, inplace=False)
	subsample.filter(
		lambda v, _, __: v.sum() > 1e-9, axis='observation', inplace=True)
	return subsample

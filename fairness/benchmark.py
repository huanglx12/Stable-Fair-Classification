import fire
import os
import os.path as path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from fairness.data.objects.list import DATASETS, get_dataset_names
from fairness.data.objects.ProcessedData import ProcessedData
from fairness.algorithms.list import ALGORITHMS
from fairness.metrics.list import get_metrics

from fairness.metrics.Accuracy import Accuracy
from collections import defaultdict
import numpy as np

from fairness.algorithms.ParamGridSearch import ParamGridSearch

NUM_TRIALS_DEFAULT = 1

def get_algorithm_names():
    result = [algorithm.get_name() for algorithm in ALGORITHMS]
    print("Available algorithms:")
    for a in result:
        print("  %s" % a)
    return result

def run(num_trials = NUM_TRIALS_DEFAULT, dataset = get_dataset_names(),
        algorithm = get_algorithm_names()):
    algorithms_to_run = algorithm

    print("Datasets: '%s'" % dataset)
    for dataset_obj in DATASETS:
        if not dataset_obj.get_dataset_name() in dataset:
            continue

        print("\nEvaluating dataset:" + dataset_obj.get_dataset_name())

        processed_dataset = ProcessedData(dataset_obj)
        # train_test_splits = processed_dataset.create_train_test_splits(num_trials)
        train_test_splits, extra_tests = processed_dataset.create_train_test_splits_and_extra_tests(num_trials)

        all_sensitive_attributes = dataset_obj.get_sensitive_attributes_with_joint()
        for sensitive in all_sensitive_attributes:

            print("Sensitive attribute:" + sensitive)

            detailed_files = dict((k, create_detailed_file(
                                          dataset_obj.get_results_filename(sensitive, k),
                                          dataset_obj,
                                          processed_dataset.get_sensitive_values(k), k))
                for k in train_test_splits.keys())

            for algorithm in ALGORITHMS:
                if not algorithm.get_name() in algorithms_to_run:
                    continue

                print("    Algorithm: %s" % algorithm.get_name())
                print("       supported types: %s" % algorithm.get_supported_data_types())
                if algorithm.__class__ is ParamGridSearch:
                    param_files =  \
                        dict((k, create_detailed_file(
                                     dataset_obj.get_param_results_filename(sensitive, k,
                                                                            algorithm.get_name()),
                                     dataset_obj, processed_dataset.get_sensitive_values(k), k))
                          for k in train_test_splits.keys())

                param_10_predictions = defaultdict(dict)
                for i in range(0, num_trials):
                    for supported_tag in algorithm.get_supported_data_types():
                        train, test = train_test_splits[supported_tag][i]
                        extra_test = extra_tests[supported_tag]
                        try:
                            params, results, param_results, param_predictions =  \
                                run_eval_alg(algorithm, train, extra_test, dataset_obj, processed_dataset,
                                             all_sensitive_attributes, sensitive, supported_tag)

                            if supported_tag not in param_10_predictions:
                                param_10_predictions[supported_tag] = {}
                            for param_val, predictions in param_predictions.items():
                                if param_val not in param_10_predictions[supported_tag]:
                                    param_10_predictions[supported_tag][param_val] = []
                                param_10_predictions[supported_tag][param_val] += [predictions]
                        except Exception as e:
                            import traceback
                            traceback.print_exc(file=sys.stderr)
                            print("Failed: %s" % e, file=sys.stderr)
                        else:
                            write_alg_results(detailed_files[supported_tag],
                                              algorithm.get_name(), params, i, results)
                            if algorithm.__class__ is ParamGridSearch:
                                for params, results in param_results:
                                    write_alg_results(param_files[supported_tag],
                                                      algorithm.get_name(), params, i, results)

                # write_own_metric(dataset_obj, algorithm, param_10_predictions)

            print("Results written to:")
            for supported_tag in algorithm.get_supported_data_types():
                print("    " + dataset_obj.get_results_filename(sensitive, supported_tag))

            for detailed_file in detailed_files.values():
                detailed_file.close()

def write_own_metric(dataset_obj, alg, param_10_predictions):
    for supported_tag, param_predictions in param_10_predictions.items():
        filename = 'own-metric_' + dataset_obj.get_dataset_name() + '_' + alg.get_name() + '_' + supported_tag + '.csv'
        f_out = open(filename, 'w')
        print('params', 'own_metric', sep='\t', file=f_out)
        for param_val, predictions in param_predictions.items():
            diff_nums = compute_diff_nums(predictions)
            print(param_val, diff_nums, sep='\t', file=f_out)
        f_out.close()

def compute_diff_nums(predictions):
    diff_nums = 0
    for i in range(len(predictions) - 1):
        prediction1 = predictions[i]
        for j in range(i+1, len(predictions)):
            prediction2 = predictions[j]
            for idx in range(len(prediction1)):
                if prediction1[idx] != prediction2[idx]:
                    diff_nums += 1
    return diff_nums

def write_alg_results(file_handle, alg_name, params, run_id, results_list):
    line = alg_name + ','
    params = ";".join("%s=%s" % (k, v) for (k, v) in params.items())
    line += params + (',%s,' % run_id)
    line += ','.join(str(x) for x in results_list) + '\n'
    file_handle.write(line)
    # Make sure the file is written to disk line-by-line:
    file_handle.flush()
    os.fsync(file_handle.fileno())

def run_eval_alg(algorithm, train, test, dataset, processed_data, all_sensitive_attributes,
                 single_sensitive, tag):
    """
    Runs the algorithm and gets the resulting metric evaluations.
    """
    privileged_vals = dataset.get_privileged_class_names_with_joint(tag)
    positive_val = dataset.get_positive_class_val(tag)

    # get the actual classifications and sensitive attributes
    actual = test[dataset.get_class_attribute()].values.tolist()
    sensitive = test[single_sensitive].values.tolist()
    print("len actual, sum:", len(actual), sum(actual))
    print("len sensitive, sum:", len(sensitive), sum(sensitive))

    predicted, params, predictions_list =  \
        run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
                privileged_vals, positive_val)
    print("len predicted, sum:", len(predicted), sum(predicted))

    # make dictionary mapping sensitive names to sensitive attr test data lists
    dict_sensitive_lists = {}
    for sens in all_sensitive_attributes:
        dict_sensitive_lists[sens] = test[sens].values.tolist()

    sensitive_dict = processed_data.get_sensitive_values(tag)
    one_run_results = []
    for metric in get_metrics(dataset, sensitive_dict, tag):
        result = metric.calc(actual, predicted, dict_sensitive_lists, single_sensitive,
                             privileged_vals, positive_val)
        one_run_results.append(result)

    # handling the set of predictions returned by ParamGridSearch
    results_lol = []
    param_predictions = {}
    if len(predictions_list) > 0:
        for param_name, param_val, predictions in predictions_list:
            params_dict = { param_name : param_val }
            results = []
            for metric in get_metrics(dataset, sensitive_dict, tag):
                result = metric.calc(actual, predictions, dict_sensitive_lists, single_sensitive,
                                     privileged_vals, positive_val)
                results.append(result)
            results_lol.append( (params_dict, results) )
            param_predictions[param_val] = predictions
    return params, one_run_results, results_lol, param_predictions

def run_alg(algorithm, train, test, dataset, all_sensitive_attributes, single_sensitive,
            privileged_vals, positive_val):
    class_attr = dataset.get_class_attribute()
    params = algorithm.get_default_params()

    # Note: the training and test set here still include the sensitive attributes because
    # some fairness aware algorithms may need those in the dataset.  They should be removed
    # before any model training is done.
    predictions, predictions_list =  \
        algorithm.run(train, test, class_attr, positive_val, all_sensitive_attributes,
                      single_sensitive, privileged_vals, params)

    return predictions, params, predictions_list

def get_metrics_list(dataset, sensitive_dict, tag):
    return [metric.get_name() for metric in get_metrics(dataset, sensitive_dict, tag)]

def get_detailed_metrics_header(dataset, sensitive_dict, tag):
    return ','.join(['algorithm', 'params', 'run-id'] + get_metrics_list(dataset, sensitive_dict, tag))

def get_dict_sensitive_vals(dict_sensitive_lists):
    """
    Takes a dictionary mapping sensitive attributes to lists in the test data and returns a
    dictionary mapping sensitive attributes to lists containing each sensitive value only once.
    """
    newdict = {}
    for sens in dict_sensitive_lists:
         sensitive = dict_sensitive_lists[sens]
         newdict[sens] = list(set(sensitive))
    return newdict

def create_detailed_file(filename, dataset, sensitive_dict, tag):
    f = open(filename, 'w')
    f.write(get_detailed_metrics_header(dataset, sensitive_dict, tag) + '\n')
    return f

def main():
    fire.Fire(run)

if __name__ == '__main__':
    main()

import pandas as pd
import numpy
import numpy.random

TAGS = ["original", "numerical", "numerical-binsensitive", "categorical-binsensitive"]
EXTRA_TEST_PERCENT = 1.0 / 5.0
TRAINING_PERCENT = 3.0 / 4.0

class ProcessedData():
    def __init__(self, data_obj):
        self.data = data_obj
        self.dfs = dict((k, pd.read_csv(self.data.get_filename(k)))
                        for k in TAGS)
        self.splits = dict((k, []) for k in TAGS)
        self.has_splits = False

        self.extra_tests = {}
        self.reserved_splits = dict((k, []) for k in TAGS)
        self.has_extra_splits = False

    def get_processed_filename(self, tag):
        return self.data.get_filename(tag)

    def get_dataframe(self, tag):
        return self.dfs[tag]

    def create_train_test_splits(self, num):
        if self.has_splits:
            return self.splits

        for i in range(0, num):
            # we first shuffle a list of indices so that each subprocessed data
            # is split consistently
            n = len(list(self.dfs.values())[0])

            a = numpy.arange(n)
            numpy.random.shuffle(a)

            split_ix = int(n * TRAINING_PERCENT)
            train_fraction = a[:split_ix]
            test_fraction = a[split_ix:]
            
            for (k, v) in self.dfs.items():
                train = self.dfs[k].iloc[train_fraction]
                test = self.dfs[k].iloc[test_fraction]
                self.splits[k].append((train, test))

        self.has_splits = True
        return self.splits

    def create_train_test_splits_and_extra_tests(self, num):
        if self.has_extra_splits:
            return self.reserved_splits, self.extra_tests

        n = len(list(self.dfs.values())[0])
        a = numpy.arange(n)
        numpy.random.shuffle(a)
        extra_split_ix = int(n * EXTRA_TEST_PERCENT)
        extra_test_fraction = a[:extra_split_ix]
        reserved_fraction = a[extra_split_ix:]
        reserved_dfs = {}

        for k, v in self.dfs.items():
            self.extra_tests[k] = self.dfs[k].iloc[extra_test_fraction]
            reserved_dfs[k] = self.dfs[k].iloc[reserved_fraction]

        for i in range(0, num):
            # we first shuffle a list of indices so that each subprocessed data
            # is split consistently
            n = len(list(reserved_dfs.values())[0])

            a = numpy.arange(n)
            numpy.random.shuffle(a)

            split_ix = int(n * TRAINING_PERCENT)
            train_fraction = a[:split_ix]
            test_fraction = a[split_ix:]

            for (k, v) in self.dfs.items():
                train = reserved_dfs[k].iloc[train_fraction]
                test = reserved_dfs[k].iloc[test_fraction]
                self.reserved_splits[k].append((train, test))

        self.has_extra_splits = True
        return self.reserved_splits, self.extra_tests

    def get_sensitive_values(self, tag):
        """
        Returns a dictionary mapping sensitive attributes in the data to a list of all possible
        sensitive values that appear.
        """
        df = self.get_dataframe(tag)
        all_sens = self.data.get_sensitive_attributes_with_joint()
        sensdict = {}
        for sens in all_sens:
             sensdict[sens] = list(set(df[sens].values.tolist()))
        return sensdict


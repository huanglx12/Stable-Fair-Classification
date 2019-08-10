This repository is meant to facilitate the benchmarking of stable and fair algorithms.

The associated paper is:

Stable and fair classification. Lingxiao Huang and Nisheeth K. Vishnoi. https://arxiv.org/abs/1902.07823.

The project is an extension of another project whose associated paper is:

A comparative study of fairness-enhancing interventions in machine learning by Sorelle A. Friedler, Carlos Scheidegger, Suresh Venkatasubramanian, Sonam Choudhary, Evan P. Hamilton, and Derek Roth. https://arxiv.org/abs/1802.04422

To install this software run:

    $ pip3 install fairness

The below instructions are still in the process of being updated to work with the new pip install-able version.

To run the benchmarks:

    $ from fairness.benchmark import run
    $ run()

This will write out metrics for each dataset to the results/ directory.

If you do not yet have all the packages installed, you may need to run:

    $ pip install -r requirements.txt

*Optional*:  The benchmarks rely on preprocessed versions of the datasets that have been included
in the repository.  If you would like to regenerate this preprocessing, run the below command
before running the benchmark script:

    $ python3 preprocess.py

Important parameter setting:

1) You can select datasets in fairness/data/objects/list.py
2) You can select algorithms in fairness/algorithms/list.py
3) The regularization term is set to be 1 by default. You can modify it in the documents of specified algorithms,
e.g., fairness/algorithms/GoelReg/GoelRegAlgorithm.py Line 83.
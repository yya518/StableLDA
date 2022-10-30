## Stable LDA  -- Extracting Actionable Insights from Text Data

### Introduction
Stable LDA is a stable topic modeling approach that generates stable model estimations. Variables generated from Stable LDA can lead to more consistent estimations in regression analyses.

In this repo, we demonstrate the use of Stable LDA in topic modeling (data exploration) and regressions (variable generation).
1. [stability_experiment](stability_experiment.ipynb) shows the use of Stable LDA for topic modeling and stability validation.
2. [stackexchange_empirical](stackexchange_empirical.ipynb) shows the use of Stable LDA on the stackexchange dataset. LDA is benchmarked as well.
3. [stackexchange_topic_modeling](stackexchange_topic_modeling.ipynb) shows the use of Stable LDA for topic modeling.


### Environment
1. Python2.7
2. gensim==3.8.3
3. scipy==1.2.3
4. scikit-learn==0.19.1
5. gcc==9.4.0 (Ubuntu) or mingw32-make (GNU Make 4.3) (Windows)

Note: the code has been tested on both Ubuntu and Windows system, but it is only tested in Python2.7, not in Python3+ yet.

### Use
For Windows user,
1. Comment line 14 in the Makefile, and Uncomment line 13 in the Makefile
2. Comment line 313 in the stablelda.py, and Uncomment line 312 in the stablelda.py
3. Run command ``mingw32-make``

For Linux user,
1. Comment line 13 in the Makefile, and Uncomment line 14 in the Makefile
2. Comment line 312 in the stablelda.py, and Uncomment line 313 in the stablelda.py
3. Run command ``make``

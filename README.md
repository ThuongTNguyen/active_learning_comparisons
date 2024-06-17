## Active Learning Benchmarking Framework

This is a framework for benchmarking *Active Learning (AL)* algorithms.
The only assumption made is this AL in the *batch* setting, i.e., you label a batch of instances
per AL iteration. Aside from that, the framework is immensely flexible, and allows plugging in various 
different components.

## Quickstart
The main AL loop is `batch_active_learn()` in `core/al.py`. 
Here are some parameters it accepts:

* `X_L`: can be None, if we want to bootstrap from the unlabeled pool.
* `y_L`: needs to be None if `X_L` is None.
* `X_U`: pool of unlabeled instances.
* `y_U`: we need this for eval, of course, in real-life these are not known.
* `X_test`: test set, against which to accuracy may be measured.
* `y_test`: labels for the test set.
* `clf_class`: the prediction model's class.
* `clf_param_grid`: scikit-type param grid for the model selection of the classifier.
* `transform_class`: can be None, if no transformation is required. This might be needed if your classifier
needs vectors, but your input `X_L` is text.
* `trf_param_grid`: scikit-type param grid for the transformation, can be None if default
    params are to be used.
* `acq_fn`: **acquisition function** or **query strategy**, usually the core of an AL algorithm, 
should accept current `X_L`, `y_L`, `X_U`, `y_U`
* `seed_size`: data to be put into the labelled set before beginning the AL loop
* `batch_size`: batch size for AL
* `num_iters`: number of times to active learn, will stop early if we exhaust the unlabeled pool
* `init_function`: how to pick the initial seed data, accepts `X_U`, `y_U`. Needs to return indices in `X_U`.
* `model_selector`: how to perform model selection at an iteration, the default is `model_selection.select_model()`
    but this can be a custom function, see `demo.py`.
* `model_search_type`: if you want to use the in-built model selectors, `cv` (for cross-validation - expensive) or 
`val` (use one validation set, cheaper and usually a good compromise).
 
Here's a simple example from `demo.py` (this is `al_demo_1()` in the file), where we perform text classification using scikit's `LinearSVC`, the model
selection is based on *cross-validation*, over these values for `C`: `[1, 0.01]`, there is a transformation to be
applied via scikit's `CountVectorizer`, and the transformation step has its own parameters - 
in this case, they are `'ngram_range': [(1, 1), (1, 3)]`. The acquisition function is *random sampling*.

```Python
import numpy as np
from core import al
from core.model_selection import select_model
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from init_strategies import init_random
from acquisition_strategies import acq_random
from sklearn.metrics import f1_score, make_scorer
from functools import partial

f1_macro = partial(f1_score, average='macro')
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
X_U, y_U = fetch_20newsgroups(subset='train', categories=categories, return_X_y=True)
X_test, y_test = fetch_20newsgroups(subset='test', categories=categories, return_X_y=True)
al.batch_active_learn(X_L=None, y_L=None, X_U=np.array(X_U), y_U=np.array(y_U),
                      X_test=np.array(X_test), y_test=np.array(y_test),
                      clf_class=LinearSVC, clf_param_grid={'C': [1, 0.01]},
                      transform_class=CountVectorizer, trf_param_grid={'ngram_range': [(1, 1), (1, 3)]},
                      acq_fn=acq_random, seed_size=20, batch_size=10, num_iters=10,
                      init_fn=init_random, model_selector=select_model,
                      model_search_type='cv', num_cv_folds=3, val_size=0.8,
                      metric=f1_macro, op_dir=r'scratch/demo_al', random_state=10)
```

[//]: # (### Citation)

[//]: # (```tex)

[//]: # (@misc{ghose2024fragility,)

[//]: # (      title={On the Fragility of Active Learners}, )

[//]: # (      author={Abhishek Ghose and Emma Thuong Nguyen},)

[//]: # (      year={2024},)

[//]: # (      eprint={2403.15744},)

[//]: # (      archivePrefix={arXiv},)

[//]: # (      primaryClass={id='cs.LG' full_name='Machine Learning' is_active=True alt_name=None in_archive='cs' is_general=False description='Papers on all aspects of machine learning research &#40;supervised, unsupervised, reinforcement learning, bandit problems, and so on&#41; including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'})

[//]: # (})

[//]: # (```)
**NOTE**: The results reported in the above article were based on the evaluation scores in 
`results/collated`.

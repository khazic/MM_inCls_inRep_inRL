# Copyright 2020 The HuggingFace Evaluate Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import evaluate
import numpy as np
from sklearn.metrics import f1_score

_DESCRIPTION = """
F1 score for multi-label classification.
"""

_KWARGS_DESCRIPTION = """
Args:
    predictions: Array of predicted labels, as returned by a model.
    references: Array of ground truth labels.
    average: String indicating the type of averaging to be performed.
        Can be one of ['micro', 'macro', 'samples', 'weighted', 'binary', None].
Returns:
    f1: F1 score.
"""

_CITATION = """\
@article{scikit-learn,
  title={Scikit-learn: Machine Learning in {P}ython},
  author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
  journal={Journal of Machine Learning Research},
  volume={12},
  pages={2825--2830},
  year={2011}
}
"""

@evaluate.utils.file_utils.add_start_docstrings(_DESCRIPTION, _KWARGS_DESCRIPTION)
class F1(evaluate.Metric):
    def _info(self):
        return evaluate.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=self._default_features,
            reference_urls=["https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html"],
            codebase_urls=["https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/metrics/_classification.py"],
            config_name="multilabel"
        )
    
    def _compute(self, predictions, references, average="micro"):
        predictions = np.array(predictions)
        references = np.array(references)
        return {
            "f1": float(f1_score(references, predictions, average=average))
        }

    @property
    def _default_features(self):
        return {
            "predictions": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
            "references": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
        }

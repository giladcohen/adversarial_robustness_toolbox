import logging
from typing import Optional, TYPE_CHECKING
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import Module

import numpy as np
import os
import time
from captum.influence import TracInCPFast

from research.utils import load_state_dict
from pytorch_influence_functions import calc_self_influence

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

logger = logging.getLogger(__name__)


class SelfInfluenceFunctionAttack(MembershipInferenceAttack):
    attack_params = MembershipInferenceAttack.attack_params + [
        "influence_score_min",
        "influence_score_max",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE", debug_dir: Optional[str] = None,
                 influence_score_min: Optional[float] = None, influence_score_max: Optional[float] = None):
        super().__init__(estimator=estimator)
        self.influence_score_min = influence_score_min
        self.influence_score_max = influence_score_max
        self.device = 'cuda'
        self.batch_size = 100
        self.debug_dir = debug_dir
        self._check_params()

    def fit(self, x_member: np.ndarray, y_member: np.ndarray, x_non_member: np.ndarray, y_non_member: np.ndarray):
        if x_member.shape[0] != x_non_member.shape[0]:
            raise ValueError("Number of members and non members do not match")
        if y_member.shape[0] != y_non_member.shape[0]:
            raise ValueError("Number of members' labels and non members' labels do not match")

        start = time.time()
        logger.info('Generating self influence scores for members (train)...')
        self_influences_member = calc_self_influence(x_member, y_member, self.estimator.model)
        end = time.time()
        logger.info('self influence scores calculation time is: {} sec'.format(end - start))

        logger.info('Generating self influence scores for non members (train)...')
        self_influences_non_member = calc_self_influence(x_non_member, y_non_member, self.estimator.model)
        if self.debug_dir is not None:
            np.save(os.path.join(self.debug_dir, 'self_influences_member.npy'), self_influences_member)
            np.save(os.path.join(self.debug_dir, 'self_influences_non_member.npy'), self_influences_non_member)

        minn = self_influences_member.min()
        maxx = self_influences_member.max()
        delta = maxx - minn
        if self.influence_score_min is None:
            self.influence_score_min = minn - delta * 0.03
        if self.influence_score_max is None:
            self.influence_score_max = maxx + delta * 0.03

        logger.info('Done fitting {}'.format(__class__))

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")
        assert y.shape[0] == x.shape[0], 'Number of rows in x and y do not match'

        logger.info('Generating self influence scores for members (infer)...')
        scores = calc_self_influence(x, y, self.estimator.model)

        if self.influence_score_min is None or self.influence_score_max is None:  # pragma: no cover
            raise ValueError(
                "No value for threshold `influence_score_min` or 'influence_score_max' provided. Please set them"
                "or run method `fit` on known training set."
            )

        y_pred = self.estimator.predict(x, self.batch_size).argmax(axis=1)
        predicted_class = np.ones(x.shape[0])  # member by default
        for i in range(x.shape[0]):
            if scores[i] < self.influence_score_min or scores[i] > self.influence_score_max:
                predicted_class[i] = 0
            if y_pred[i] != y[i]:
                predicted_class[i] = 0

        return predicted_class

    def _check_params(self) -> None:
        if self.influence_score_min is not None and not isinstance(self.influence_score_min, (int, float)):
            raise ValueError("The influence threshold `influence_score_min` needs to be a float.")
        if self.influence_score_max is not None and not isinstance(self.influence_score_max, (int, float)):
            raise ValueError("The influence threshold `influence_score_max` needs to be a float.")
        if self.influence_score_max is not None and self.influence_score_min is not None and (self.influence_score_max <= self.influence_score_min):
            raise ValueError("This is mandatory: influence_score_min < influence_score_max")

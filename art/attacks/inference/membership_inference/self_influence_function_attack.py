import logging
from typing import Optional, TYPE_CHECKING
import torch
from torch.utils.data import TensorDataset
import torch.nn as nn
from torch.nn import Module
import warnings
import numpy as np
import os
import time
from tqdm import tqdm
from sklearn.metrics import roc_curve

from research.consts import RGB_MEAN, RGB_STD
from research.utils import load_state_dict, save_to_path, normalize
from pytorch_influence_functions import calc_self_influence, calc_self_influence_adaptive, \
    calc_self_influence_average, calc_self_influence_adaptive_for_ref, calc_self_influence_average_for_ref, \
    calc_self_influence_for_ref

from art.attacks.attack import MembershipInferenceAttack
from art.estimators.estimator import BaseEstimator
from art.estimators.classification.classifier import ClassifierMixin
from art.utils import check_and_transform_label_format

if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE


#suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class SelfInfluenceFunctionAttack(MembershipInferenceAttack):
    attack_params = MembershipInferenceAttack.attack_params + [
        "influence_score_min",
        "influence_score_max",
    ]
    _estimator_requirements = (BaseEstimator, ClassifierMixin)

    def __init__(self, estimator: "CLASSIFIER_TYPE", debug_dir: str, miscls_as_nm: bool = True, adaptive: bool = False,
                 average: bool = False, for_ref: bool = False, rec_dep: int = 1, r: int = 1,
                 optimize_tpr_fpr: bool = False, influence_score_min: Optional[float] = None,
                 influence_score_max: Optional[float] = None):
        super().__init__(estimator=estimator)
        self.influence_score_min = influence_score_min
        self.influence_score_max = influence_score_max
        self.device = 'cuda'
        self.miscls_as_nm = miscls_as_nm
        self.adaptive = adaptive
        self.average = average
        self.for_ref = for_ref
        self.rec_dep = rec_dep
        self.r = r
        self.optimize_tpr_fpr = optimize_tpr_fpr
        self.batch_size = 100
        self.num_fit_iters = 20
        self.threshold_bins: list = []
        self.debug_dir = debug_dir
        self.self_influences_member_train_path = os.path.join(self.debug_dir, 'self_influences_member_train.npy')
        self.self_influences_non_member_train_path = os.path.join(self.debug_dir, 'self_influences_non_member_train.npy')
        self.self_influences_member_test_path = os.path.join(self.debug_dir, 'self_influences_member_test.npy')
        self.self_influences_non_member_test_path = os.path.join(self.debug_dir, 'self_influences_non_member_test.npy')
        self._check_params()

        if self.adaptive:
            if self.for_ref:
                self.self_influence_func = calc_self_influence_adaptive_for_ref
                logger.info('Setting self influence attack with adaptive attack suited for ref paper')
            else:
                self.self_influence_func = calc_self_influence_adaptive
                logger.info('Setting self influence attack with adaptive attack')
        elif self.average:
            if self.for_ref:
                self.self_influence_func = calc_self_influence_average_for_ref
                logger.info('Setting self influence attack with ensemble attack suited for ref paper')
            else:
                self.self_influence_func = calc_self_influence_average
                logger.info('Setting self influence attack with ensemble attack')
        else:
            if self.for_ref:
                self.self_influence_func = calc_self_influence_for_ref
                logger.info('Setting self influence attack with vanilla attack for ref paper')
            else:
                self.self_influence_func = calc_self_influence
                logger.info('Setting self influence attack with vanilla attack')

    def fit(self, x_member: np.ndarray, y_member: np.ndarray, x_non_member: np.ndarray, y_non_member: np.ndarray):
        assert x_member.shape[0] == y_member.shape[0], 'Number of rows in x and y do not match for members'
        assert x_non_member.shape[0] == y_non_member.shape[0], 'Number of rows in x and y do not match for non-members'

        start = time.time()
        assert os.path.exists(self.self_influences_member_train_path)  # debug for only inference results
        assert os.path.exists(self.self_influences_non_member_train_path)  # debug for only inference results
        if os.path.exists(self.self_influences_member_train_path):
            logger.info('Loading self influence scores for members (train)...')
            self_influences_member = np.load(self.self_influences_member_train_path)
        else:
            logger.info('Generating self influence scores for members (train)...')
            self_influences_member = self.self_influence_func(x_member, y_member, self.estimator.model, self.rec_dep, self.r)
            np.save(self.self_influences_member_train_path, self_influences_member)

        if os.path.exists(self.self_influences_non_member_train_path):
            logger.info('Loading self influence scores for non members (train)...')
            self_influences_non_member = np.load(self.self_influences_non_member_train_path)
        else:
            logger.info('Generating self influence scores for non members (train)...')
            self_influences_non_member = self.self_influence_func(x_non_member, y_non_member, self.estimator.model, self.rec_dep, self.r)
            np.save(self.self_influences_non_member_train_path, self_influences_non_member)

        if self.for_ref:
            x_member = normalize(x_member, RGB_MEAN, RGB_STD)
            x_non_member = normalize(x_non_member, RGB_MEAN, RGB_STD)

        y_pred_member = self.estimator.predict(x_member, self.batch_size).argmax(axis=1)
        y_pred_non_member = self.estimator.predict(x_non_member, self.batch_size).argmax(axis=1)
        # pred_member_mismatch = y_pred_member != y_member
        # pred_non_member_mismatch = y_pred_non_member != y_non_member
        pred_member_match = y_pred_member == y_member
        pred_non_member_match = y_pred_non_member == y_non_member

        logger.info('Fitting min and max thresholds...')
        minn = self_influences_member.min()
        maxx = self_influences_member.max()
        delta = maxx - minn
        # setting array of min/max thresholds
        minn_arr = np.linspace(minn - delta * 0.5, minn + delta * 0.5, 100)
        maxx_arr = np.linspace(maxx - delta * 0.5, maxx + delta * 0.5, 100)

        score_max = 0.0
        best_min = -np.inf
        best_max = np.inf
        if self.optimize_tpr_fpr:
            scores = np.concatenate((self_influences_non_member, self_influences_member))
            y_pred = np.concatenate((y_pred_non_member, y_pred_member))
            y = np.concatenate((y_non_member, y_member))
            y_is_member = np.concatenate((np.zeros(len(self_influences_non_member)), np.ones(len(self_influences_member))))
            found = False
            for i in tqdm(range(len(minn_arr))):
                for j in range(len(maxx_arr)):
                    tau1 = minn_arr[i]
                    tau2 = maxx_arr[j]
                    prob_1 = np.nan * np.ones_like(scores)
                    for k in range(scores.shape[0]):
                        look_left = np.abs(scores[k] - tau1) < np.abs(scores[k] - tau2)
                        if look_left:
                            dist = scores[k] - tau1
                        else:
                            dist = tau2 - scores[k]
                        prob_1[k] = 1 / (1 + np.exp(-dist))
                        if y_pred[k] != y[k] and self.miscls_as_nm:
                            prob_1[k] = 0.0

                    # calculate ROC curve
                    fpr, tpr, thresholds = roc_curve(y_is_member, prob_1)
                    # verify that fpr has at least one element between 0.0005 and 0.002:
                    valid_indices = np.where(np.logical_and(fpr >= 0.0005, fpr <= 0.002))[0]
                    if len(valid_indices) == 0:
                        # cannot find fpr close enough for 0.001
                        continue
                    else:
                        tpr_at_fpr_0p001 = tpr[np.argmin(np.abs(fpr - 0.001))]
                    if tpr_at_fpr_0p001 > score_max:
                        best_min, best_max = tau1, tau2
                        score_max = tpr_at_fpr_0p001
                        found = True
            assert found, 'Did not find FPR close to 0.001'
        else:
            self.threshold_bins = []
            for i in tqdm(range(len(minn_arr))):
                for j in range(len(maxx_arr)):
                    if self.miscls_as_nm:
                        inferred_member = np.int_(
                            np.logical_and.reduce([self_influences_member > minn_arr[i], self_influences_member < maxx_arr[j], pred_member_match])
                        )
                        inferred_non_member = np.int_(
                            np.logical_and.reduce([self_influences_non_member > minn_arr[i], self_influences_non_member < maxx_arr[j], pred_non_member_match])
                        )
                    else:
                        inferred_member = np.int_(
                            np.logical_and.reduce([self_influences_member > minn_arr[i], self_influences_member < maxx_arr[j]])
                        )
                        inferred_non_member = np.int_(
                            np.logical_and.reduce([self_influences_non_member > minn_arr[i], self_influences_non_member < maxx_arr[j]])
                        )
                    member_acc = np.mean(inferred_member == 1)
                    non_member_acc = np.mean(inferred_non_member == 0)
                    acc = (member_acc * len(inferred_member) + non_member_acc * len(inferred_non_member)) / (len(inferred_member) + len(inferred_non_member))
                    self.threshold_bins.append((minn_arr[i], maxx_arr[j], acc))
                    if acc > score_max:
                        best_min, best_max = minn_arr[i], maxx_arr[j]
                        score_max = acc

        self.influence_score_min = best_min
        self.influence_score_max = best_max

        end = time.time()
        logger.info('Fitting self influence scores calculation time is: {} sec'.format(end - start))
        logger.info('Done fitting {}'.format(__class__))

    def infer(self, x: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        if y is None:  # pragma: no cover
            raise ValueError("Argument `y` is None, but this attack requires true labels `y` to be provided.")
        assert y.shape[0] == x.shape[0], 'Number of rows in x and y do not match'

        if self.influence_score_min is None or self.influence_score_max is None:  # pragma: no cover
            raise ValueError(
                "No value for threshold `influence_score_min` or 'influence_score_max' provided. Please set them"
                "or run method `fit` on known training set."
            )

        if "probabilities" in kwargs.keys():
            probabilities = kwargs.get("probabilities")
        else:
            probabilities = False

        infer_set = kwargs.get('infer_set', None)
        assert infer_set is not None, "infer() must be called with kwargs with 'infer_set'"
        if infer_set == 'member_test':
            infer_path = self.self_influences_member_test_path
        elif infer_set == 'non_member_test':
            infer_path = self.self_influences_non_member_test_path
        else:
            raise AssertionError('Invalid value infer_set = {}'.format(infer_set))

        if os.path.exists(infer_path):
            logger.info('Loading self influence scores from {} (infer)...'.format(infer_path))
            scores = np.load(infer_path)
        else:
            logger.info('Generating self influence scores to {} (infer)...'.format(infer_path))
            scores = self.self_influence_func(x, y, self.estimator.model, self.rec_dep, self.r)
            np.save(infer_path, scores)

        if self.for_ref:
            x = normalize(x, RGB_MEAN, RGB_STD)
        y_pred = self.estimator.predict(x, self.batch_size).argmax(axis=1)

        if not probabilities:
            predicted_class = np.ones(x.shape[0])  # member by default
            for i in range(x.shape[0]):
                if scores[i] < self.influence_score_min or scores[i] > self.influence_score_max:
                    predicted_class[i] = 0
                if y_pred[i] != y[i] and self.miscls_as_nm:
                    predicted_class[i] = 0

            return predicted_class
        else:
            assert self.influence_score_min is not None and self.influence_score_max is not None
            prob_1 = np.nan * np.ones_like(scores)
            for i in range(x.shape[0]):
                look_left = np.abs(scores[i] - self.influence_score_min) < np.abs(scores[i] - self.influence_score_max)
                if look_left:
                    dist = scores[i] - self.influence_score_min
                else:
                    dist = self.influence_score_max - scores[i]
                prob_1[i] = 1 / (1 + np.exp(-dist))
                if y_pred[i] != y[i] and self.miscls_as_nm:
                    prob_1[i] = 0.0
            # prob_0 = np.ones_like(prob_1) - prob_1
            # probs = np.stack((prob_0, prob_1), axis=1)
            return prob_1

    def _check_params(self) -> None:
        if not (isinstance(self.rec_dep, int) and self.rec_dep >= 1):
            raise ValueError("The argument `rec_dep` needs to be an int, and not lower than 1.")
        if not (isinstance(self.r, int) and self.r >= 1):
            raise ValueError("The argument `r` needs to be an int, and not lower than 1.")
        if self.influence_score_min is not None and not isinstance(self.influence_score_min, (int, float)):
            raise ValueError("The influence threshold `influence_score_min` needs to be a float.")
        if self.influence_score_max is not None and not isinstance(self.influence_score_max, (int, float)):
            raise ValueError("The influence threshold `influence_score_max` needs to be a float.")
        if self.influence_score_max is not None and self.influence_score_min is not None and (self.influence_score_max <= self.influence_score_min):
            raise ValueError("This is mandatory: influence_score_min < influence_score_max")
        if self.adaptive + self.average > 1:
            raise ValueError("Can only set one of self.adaptive, self.average to True.")

"""The feature transform and classifier shared by training and prediction."""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

N_READOUT = 6000


def select_columns(features, n_neurons, bins, limit=N_READOUT):
    totals = sum(features[:, b * n_neurons:(b + 1) * n_neurons] for b in range(bins))
    mean = np.asarray(totals.mean(axis=0)).ravel()
    variance = np.asarray(totals.multiply(totals).mean(axis=0)).ravel() - mean ** 2
    selected = np.argsort(-variance, kind='stable')[:min(limit, n_neurons)]
    return np.concatenate([b * n_neurons + selected for b in range(bins)])


def transform(features, columns=None):
    if columns is None:
        return np.asarray(features, dtype=np.float32)
    values = features[:, columns]
    if hasattr(values, 'toarray'):
        values = values.toarray()
    return np.log1p(np.asarray(values, dtype=np.float32))


def fit(features, labels, columns=None):
    values = transform(features, columns)
    scaler = StandardScaler()
    values = scaler.fit_transform(values)
    classifier = LogisticRegression(C=0.05, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        classifier.fit(values, labels)
    return {
        'cols': columns,
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'coef': classifier.coef_,
        'intercept': classifier.intercept_,
        'classes': classifier.classes_,
    }


def probabilities(model, features):
    values = transform(features, model['cols'])
    scores = ((values - model['mean']) / model['scale']) @ model['coef'].T + model['intercept']
    scores -= scores.max(axis=1, keepdims=True)
    scores = np.exp(scores)
    return scores / scores.sum(axis=1, keepdims=True)


def predict(model, features):
    return model['classes'][probabilities(model, features).argmax(axis=1)]

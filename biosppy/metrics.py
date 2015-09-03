﻿# -*- coding: utf-8 -*-
"""
    biosppy.metrics
    ---------------
    
    This module provides pairwise distance computation methods.
    
    :copyright: (c) 2015 by Instituto de Telecomunicacoes
    :license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in

# 3rd party
import numpy as np
import scipy.spatial.distance as ssd
from scipy import linalg

# local

# Globals


def pcosine(u, v):
    """Computes the Cosine distance (positive space) between 1-D arrays.
    
    The Cosine distance (positive space) between `u` and `v` is defined as
    
    .. math::
    
        1 - \\abs(\\frac{u \\cdot v}{||v||_2 ||v||_2})
    
    where :math:`u \\cdot v` is the dot product of :math:`u` and :math:`v`.
    
    Args:
        u (array): Input array.
        
        v (array): Input array.
    
    Returns:
        cosine (float): Cosine distance between `u` and `v`.
    
    """
    
    # validate vectors like scipy does
    u = ssd._validate_vector(u)
    v = ssd._validate_vector(v)
    
    dist = 1. - np.abs(np.dot(u, v) / (linalg.norm(u) * linalg.norm(v)))
    
    return dist


def pdist(X, metric='euclidean', p=2, w=None, V=None, VI=None):
    """Pairwise distances between observations in n-dimensional space.
    
    Wraps scipy.spatial.distance.pdist.
    
    Args:
        X (array): An m by n array of m original observations in an
        n-dimensional space.
        
        metric (str, function): The distance metric to use; the distance can be
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
            'mahalanobis', 'matching', 'minkowski', 'pcosine', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            'sqeuclidean', 'yule'.
        
        w (array): The weight vector (for weighted Minkowski).
        
        p (float): The p-norm to apply (for Minkowski, weighted and unweighted).
        
        V (array): The variance vector (for standardized Euclidean).
        
        VI (array): The inverse of the covariance matrix (for Mahalanobis).
    
    Returns:
        Y (array): Returns a condensed distance matrix Y.  For each :math:`i`
            and :math:`j` (where :math:`i<j<n`), the metric
            ``dist(u=X[i], v=X[j])`` is computed and stored in entry ``ij``.
    
    """
    
    if isinstance(metric, basestring):
        if metric == 'pcosine':
            metric = pcosine
    
    return ssd.pdist(X, metric, p, w, V, VI)


def cdist(XA, XB, metric='euclidean', p=2, V=None, VI=None, w=None):
    """Computes distance between each pair of the two collections of inputs.
    
    Wraps scipy.spatial.distance.cdist.
    
    Args:
        XA (array): An :math:`m_A` by :math:`n` array of :math:`m_A` original
            observations in an :math:`n`-dimensional space.
        
        XB (array): An :math:`m_B` by :math:`n` array of :math:`m_B` original
            observations in an :math:`n`-dimensional space.
        
        metric (str, function): The distance metric to use; the distance can be
            'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
            'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
            'mahalanobis', 'matching', 'minkowski', 'pcosine', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
            'sqeuclidean', 'yule'.
        
        w (array): The weight vector (for weighted Minkowski).
        
        p (float): The p-norm to apply (for Minkowski, weighted and unweighted).
        
        V (array): The variance vector (for standardized Euclidean).
        
        VI (array): The inverse of the covariance matrix (for Mahalanobis).
    
    Returns:
        Y (array): A :math:`m_A` by :math:`m_B` distance matrix is returned.
            For each :math:`i` and :math:`j`, the metric
            ``dist(u=XA[i], v=XB[j])`` is computed and stored in
            the :math:`ij` th entry.
    
    """
    
    if isinstance(metric, basestring):
        if metric == 'pcosine':
            metric = pcosine
    
    return ssd.cdist(XA, XB, metric, p, V, VI, w)


def squareform(X, force="no", checks=True):
    """Converts a vector-form distance vector to a square-form distance matrix,
    and vice-versa.
    
    Wraps scipy.spatial.distance.squareform.
    
    Args:
        X (array): Either a condensed or redundant distance matrix.
        
        force (str): As with MATLAB(TM), if force is equal to 'tovector' or
            'tomatrix', the input will be treated as a distance matrix or
            distance vector respectively.
        
        checks (bool): If `checks` is set to False, no checks will be made for
            matrix symmetry nor zero diagonals. This is useful if it is known
            that ``X - X.T1`` is small and ``diag(X)`` is close to zero. These
            values are ignored any way so they do not disrupt the squareform
            transformation.
    
    Returns:
        Y (array): If a condensed distance matrix is passed, a redundant one is
            returned, or if a redundant one is passed, a condensed distance
            matrix is returned.
    
    """
    
    return ssd.squareform(X, force, checks)

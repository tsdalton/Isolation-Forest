import numpy as np
from iforest import IsolationForest

class IFWrapper(object):
    """Wrapper class for the Isolation Forest model.

    Return the anomaly score of each sample with the IsolationForest algorithm

    IsolationForest consists in 'isolate' the observations by randomly
    selecting a feature and then randomly selecting a split value
    between the maximum and minimum values of the selected feature.

    Since recursive partitioning can be represented by a tree structure, the
    number of splitting required to isolate a point is equivalent to the path
    length from the root node to a terminating node.

    This path length, averaged among a forest of such random trees, is a
    measure of abnormality and our decision function.

    Indeed random partitioning produces noticeable shorter paths for anomalies.
    Hence, when a forest of random trees collectively produce shorter path
    lengths for some particular points, then they are highly likely to be
    anomalies.


    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of base estimators in the ensemble.

    max_samples : int or float, optional (default=256)
        The number of samples to draw from X to train each base estimator.
            - If int, then draw `max_samples` samples.
            - If float, then draw `max_samples * X.shape[0]` samples.

    max_features : int or float, optional (default=1.0)
        The number of features to draw from X to train each base estimator.
            - If int, then draw `max_features` features.
            - If float, then draw `max_features * X.shape[1]` features.

    bootstrap : boolean, optional (default=True)
        Whether samples are drawn with replacement.

    n_jobs : integer, optional (default=1)
        The number of jobs to run in parallel for both `fit` and `predict`.
        If -1, then the number of jobs is set to the number of cores.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    verbose : int, optional (default=0)
        Controls the verbosity of the tree building process.


    Attributes
    ----------
    estimators_ : list of DecisionTreeClassifier
        The collection of fitted sub-estimators.

    estimators_samples_ : list of arrays
        The subset of drawn samples (i.e., the in-bag samples) for each base
        estimator.

    References
    ----------
    .. [1] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation forest."
           Data Mining, 2008. ICDM'08. Eighth IEEE International Conference on.
    .. [2] Liu, Fei Tony, Ting, Kai Ming and Zhou, Zhi-Hua. "Isolation-based
           anomaly detection." ACM Transactions on Knowledge Discovery from
           Data (TKDD) 6.1 (2012): 3.

    """

    def __init__(self,n_estimators=100, max_samples=256):
        self.model = IsolationForest(n_estimators, max_samples)
        self.threshold = 0.6 ## Recommended threshold in IForest paper.
        self.trainedStatus = False
    
    def train(self, data):
        """ Trains the model with data.
        
        Parameters
        ----------
        data : array-like or sparse matrix, shape=(n_samples, n_features)
            The input samples used for training. 
            Use ``dtype=np.float32`` for maximum efficiency. Sparse matrices 
            are also supported, use sparse ``csc_matrix`` for maximum efficieny.

        Returns
        -------
        none
        """
        self.model.fit(data)
        self.trainingData = data
        self.trainedStatus = True
        
    def getAnomScore(self, data):
        """ Returns the anomaly score of data. 
        
        Parameters
        ----------
        data : array-like or sparse matrix, shape=(n_samples, n_features) or 
            single point. The input samples. Use ``dtype=np.float32`` for 
            maximum efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficieny.    

        Returns
        -------
        scores : array of shape (n_samples,)
            The anomaly score of the input samples.
            The lower, the more normal.
        """
        data = np.asarray(data) 
        if data.shape == (2,):     # Check if single point or if array of points
            data = data.reshape(1,-1)
        return self.model.predict(data)
        
    def setThreshold(self, data, percentile):
        """ Sets the anomaly threshold for the model.
        
        Parameters
        ----------
        data : array-like or sparse matrix, shape=(n_samples, n_features).
        The input samples. Use ``dtype=np.float32`` for 
            maximum efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficieny.
            
        Percentile : floating point number. The percentile point desired. 

        Returns
        -------
        none
        """
        scores = []
        for point in data:
            scores.append(self.getAnomScore(point))
        self.threshold = np.precentile(scores, percentile)
        
    def getTopPercent(self, data, N=1.0):
        """ Returns the top N percent of anomalies. Default is top 1 percent.
        
        Parameters
        ----------
        data : array-like or sparse matrix, shape=(n_samples, n_features).
        The input samples. Use ``dtype=np.float32`` for 
            maximum efficiency. Sparse matrices are also supported, use sparse
            ``csc_matrix`` for maximum efficieny.
            
        N : floating point number. The percentage point desired. 

        Returns
        -------
        scoreIndex : tuple of index location of the top N percent anomaly scores
        """
        scores = []
        for point in data:
            point = point.reshape(1,-1)
            scores.append(self.getAnomScore(point))
        thresh = np.percentile(scores, 100.00 - N)
        return np.where(scores >= thresh)
    
    def getThreshold(self):
        """ Returns the current threshold for the model.
        
        Parameters
        ----------
        none
        
        Returns
        -------
        threshold : the model's current threshold
        """
        return self.threshold
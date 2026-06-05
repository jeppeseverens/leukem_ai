import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class DESeq2RatioNormalizer(BaseEstimator, TransformerMixin):
    """
    Transformer implementing the DESeq2 median-of-ratios normalization.

    This normalizer:
      1. Filters out lowly expressed genes based on a threshold.
      2. Computes gene-wise log counts.
      3. Normalizes raw count data using size factors derived from the median log ratio.
      4. Applies a log2 transformation (after adding a pseudocount of 1).

    Attributes
    ----------
    logmeans_ : np.ndarray
        Gene-wise mean log counts computed on filtered data.
    ok_expressed_genes_ : np.ndarray (bool)
        Boolean mask for genes that are sufficiently expressed.
    finite_genes_ : np.ndarray (bool)
        Boolean mask for genes with finite (non -∞) log means.
    """

    def fit(self, X, y=None):
        """
        Fit the normalizer by computing gene-wise log means on filtered data.

        Genes are filtered out if they do not have sufficient expression:
          - Must have non-negligible total reads.
          - Must have counts greater than 9 in at least half of the samples.

        Parameters
        ----------
        X : np.ndarray
            Raw count data (rows are samples and columns are genes).
        y : Ignored

        Returns
        -------
        self : object
            Fitted transformer.
        """
        if not isinstance(X, np.ndarray):
            raise ValueError("Input count data must be in a numpy array.")
        
        # Filter genes with a sum of reads above 10% of number of samples.
        # ok_expressed_genes = X.sum(axis=0) > (0.1 * X.shape[0])

        # Filter the array to only include expressed genes.
        # X = X[:, ok_expressed_genes]

        # Compute the natural logarithm of counts (suppress warnings for log(0)).
        with np.errstate(divide="ignore"):
            log_counts = np.log(X)

        # Compute the mean log counts for each gene.
        logmeans = log_counts.mean(axis=0)
        # Identify genes that yield finite log means.
        finite_genes = ~np.isinf(logmeans)

        # Store the computed attributes.
        self.logmeans_ = logmeans
        # self.ok_expressed_genes_ = ok_expressed_genes
        self.finite_genes_ = finite_genes
        return self

    def transform(self, X, y=None):
        """
        Normalize the raw count data using the DESeq2 median-of-ratios method.

        Steps:
          1. Filter the data to include only expressed genes.
          2. Compute log-transformed counts.
          3. Calculate log ratios by subtracting precomputed gene-wise log means.
          4. Derive sample-specific size factors using the median of log ratios.
          5. Normalize counts using the size factors and apply a log2 transformation.

        Parameters
        ----------
        X : np.ndarray
            Raw count data (rows are samples and columns are genes).
        y : Ignored

        Returns
        -------
        np.ndarray
            Normalized and log2-transformed count data.
        """
        # Select only the expressed genes determined during fitting.
        #X = X[:, self.ok_expressed_genes_]
        with np.errstate(divide="ignore"):
            log_counts = np.log(X)
        # Compute log ratios for genes with finite log means.
        log_ratios = log_counts[:, self.finite_genes_] - self.logmeans_[self.finite_genes_]
        # Calculate the median log ratio for each sample.
        log_medians = np.median(log_ratios, axis=1)
        size_factors = np.exp(log_medians)
        # Normalize each sample by its corresponding size factor.
        X = X / size_factors[:, np.newaxis]
        # Return log2-transformed normalized counts (adding 1 to avoid log2(0)).
        return np.log2(X + 1)

class FeatureSelection(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting top variable genes across multiple studies using MAD.

    This transformer selects the top 'n_genes' per study based on the median absolute deviation (MAD)
    and then uses the intersection of these genes across predefined studies.
    """
    def __init__(self):
        self.study_per_patient = None
        self.n_genes = None

    def _compute_top_genes(self, X_arr):
        # Compute MAD in a single vectorized operation
        # Subtract median and take absolute value in one step
        mad = np.median(np.abs(X_arr - np.median(X_arr, axis=0)), axis=0)
        
        # Use argpartition instead of argsort for better performance
        # This is O(n) instead of O(n log n) for full sorting
        top_indices = np.argpartition(mad, -self.n_genes)[-self.n_genes:]
        
        # Convert to set for faster intersection operations later
        return set(top_indices)

    def fit(self, X, y=None, study_per_patient=None, n_genes = 2000):
        self.n_genes = n_genes
        self.study_per_patient = study_per_patient

        if self.study_per_patient is None:
            raise ValueError("study_per_patient must be provided.")
        
        selected_studies = np.unique(study_per_patient)
        top_genes_by_study = {}
        
        for study in selected_studies:
            mask = self.study_per_patient == study
            if mask.sum() == 0:
                continue
            X_study_arr = X[mask, :]
            top_genes_by_study[study] = self._compute_top_genes(X_study_arr)

        # Compute the intersection of top genes and preserve the original order.
        intersect_genes = set.intersection(*top_genes_by_study.values())
        self.mvgs_ = [i for i in intersect_genes]
        
        return self

    def transform(self, X, y=None):
        """
        Reduce the dataset to only include the selected genes.
        """
        return X[:, self.mvgs_]

class FeatureSelection2(BaseEstimator, TransformerMixin):
    """
    Transformer for selecting top variable genes across multiple studies using MAD.

    This transformer selects the top 'n_genes' per study based on the median absolute deviation (MAD)
    and then uses the intersection of these genes across predefined studies.
    """
    def __init__(self):
        self.study_per_patient = None
        self.n_genes = None

    def _compute_top_genes(self, X_arr):
        # Pre-allocate arrays for better memory efficiency
        n_samples, n_genes = X_arr.shape
        medians = np.empty(n_genes, dtype=X_arr.dtype)
        mad = np.empty(n_genes, dtype=X_arr.dtype)
        
        # Compute medians in one pass
        np.median(X_arr, axis=0, out=medians)
        
        # Compute MAD in one pass using in-place operations
        # Subtract median and take absolute value in one step
        np.abs(X_arr - medians, out=X_arr)  # In-place operation
        np.median(X_arr, axis=0, out=mad)  # In-place operation
        
        # Use argpartition for better performance
        kth = n_genes - self.n_genes
        top_indices = np.argpartition(mad, kth)[kth:]
        
        # Convert to set for faster intersection operations later
        return set(top_indices)

    def fit(self, X, y=None, study_per_patient=None, n_genes = 2000):
        self.n_genes = n_genes
        self.study_per_patient = study_per_patient

        if self.study_per_patient is None:
            raise ValueError("study_per_patient must be provided.")
        
        selected_studies = np.unique(study_per_patient)
        top_genes_by_study = {}
        
        # Pre-allocate memory for study masks
        study_masks = {}
        for study in selected_studies:
            study_masks[study] = self.study_per_patient == study
        
        for study in selected_studies:
            mask = study_masks[study]
            if mask.sum() == 0:
                continue
            X_study_arr = X[mask, :]
            top_genes_by_study[study] = self._compute_top_genes(X_study_arr)

        # Compute the intersection of top genes and preserve the original order
        intersect_genes = set.intersection(*top_genes_by_study.values())
        self.mvgs_ = [i for i in intersect_genes]
        
        return self

    def transform(self, X, y=None):
        """
        Reduce the dataset to only include the selected genes.
        """
        return X[:, self.mvgs_] 


class FeatureSelectionMADGlobal(BaseEstimator, TransformerMixin):
    """
    Select top n_genes by MAD over all training samples (cohort-agnostic).

    Unlike FeatureSelection/FeatureSelection2, genes are ranked by variability
    in the pooled training matrix without per-study intersection.
    """

    def __init__(self):
        self.n_genes = None

    def fit(self, X, y=None, study_per_patient=None, n_genes=2000):
        self.n_genes = min(n_genes, X.shape[1])
        mad = np.median(np.abs(X - np.median(X, axis=0)), axis=0)
        kth = X.shape[1] - self.n_genes
        self.mvgs_ = list(np.argpartition(mad, kth)[kth:])
        return self

    def transform(self, X, y=None):
        return X[:, self.mvgs_]


class FeatureSelectionEta(BaseEstimator, TransformerMixin):
    """
    Transformer that selects genes based on eta-squared difference:
    eta2_subtype - eta2_study.

    For each gene, it computes the proportion of variance in expression
    explained by subtype labels and by study labels, then scores genes
    by eta2_subtype - eta2_study and keeps the top n_genes.
    """

    def __init__(self):
        self.study_per_patient = None
        self.n_genes = None

    def _eta2(self, X, groups):
        """
        Compute eta-squared per column of X for a categorical grouping vector.

        X: np.ndarray of shape (n_samples, n_genes)
        groups: array-like of length n_samples
        """
        n_samples, n_genes = X.shape
        groups = np.asarray(groups)

        overall_mean = X.mean(axis=0)
        ss_total = ((X - overall_mean) ** 2).sum(axis=0)
        # Avoid division by zero for constant genes
        ss_total[ss_total == 0] = np.nan

        ss_between = np.zeros(n_genes, dtype=X.dtype)
        for g in np.unique(groups):
            mask = groups == g
            n_g = mask.sum()
            if n_g == 0:
                continue
            mean_g = X[mask].mean(axis=0)
            ss_between += n_g * (mean_g - overall_mean) ** 2

        eta2 = ss_between / ss_total
        return eta2

    def fit(self, X, y=None, study_per_patient=None, n_genes=2000):
        """
        Fit by computing eta2_subtype and eta2_study and selecting top genes
        by (eta2_subtype - eta2_study).

        X: samples x genes matrix (numpy array)
        y: subtype labels (encoded)
        study_per_patient: array-like of study labels
        """
        if y is None:
            raise ValueError("Subtype labels (y) must be provided.")
        if study_per_patient is None:
            raise ValueError("study_per_patient must be provided.")

        self.n_genes = n_genes
        self.study_per_patient = np.asarray(study_per_patient)

        # Compute eta-squared for subtype and study
        eta2_subtype = self._eta2(X, y)
        eta2_study = self._eta2(X, self.study_per_patient)

        # Score genes: high subtype signal, low study signal
        score = eta2_subtype - eta2_study

        # Handle NaNs (e.g. constant genes): push them to the bottom
        score = np.where(np.isnan(score), -np.inf, score)

        # Select top n_genes by score
        n_genes = min(self.n_genes, X.shape[1])
        if n_genes <= 0:
            raise ValueError("n_genes must be positive.")

        top_indices = np.argpartition(score, -n_genes)[-n_genes:]
        # Keep as Python list of column indices
        self.mvgs_ = list(top_indices)

        return self

    def transform(self, X, y=None):
        """
        Reduce the dataset to only include the selected genes.
        """
        return X[:, self.mvgs_]
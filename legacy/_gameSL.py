import numpy as np
from sklearn.utils import check_random_state
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.spatial import distance

class GAME_GEN:
    def __init__(self, strategy_param=0.5, lambda_penalty=0.3, beta_f1_weight=0.5, 
                 noise_level=0.05, n_neighbors=5, feature_weights=None, 
                 max_iterations=15, adaptive_lr=0.01, pca_components=None, random_state=None):
        self.strategy_param = strategy_param
        self.lambda_penalty = lambda_penalty
        self.beta_f1_weight = beta_f1_weight
        self.noise_level = noise_level
        self.n_neighbors = n_neighbors
        self.feature_weights = feature_weights
        self.max_iterations = max_iterations  # Added limit to iterations
        self.adaptive_lr = adaptive_lr
        self.pca_components = pca_components
        self.random_state = random_state
        self._random_state = check_random_state(self.random_state)
        self.pca_model = None
        self.scaler = StandardScaler()
        
    def _find_neighbors(self, sample, data, k):
        """Find k nearest neighbors of a sample in the data."""
        # Use vectorized operations for efficiency
        distances = np.linalg.norm(data - sample, axis=1)
        # Limit k to avoid index out of bounds
        k = min(k, len(data) - 1)
        if k <= 0:
            k = 1
        neighbors_indices = np.argsort(distances)[1:k+1]  # exclude self
        return neighbors_indices
        
    def generator(self, minority_samples, num_samples=100):
        """Generate synthetic samples with enhanced strategies."""
        if len(minority_samples) <= 1:
            # Can't generate new samples with just one reference point
            return minority_samples.copy()
            
        n_features = minority_samples.shape[1]
        
        # Apply feature weighting if specified
        if self.feature_weights is None:
            self.feature_weights = np.ones(n_features)
            
        # Apply PCA transformation if specified
        use_pca = False
        if self.pca_components is not None and self.pca_components < n_features:
            try:
                if self.pca_model is None:
                    self.pca_model = PCA(n_components=self.pca_components, random_state=self.random_state)
                    transformed_samples = self.pca_model.fit_transform(minority_samples)
                else:
                    transformed_samples = self.pca_model.transform(minority_samples)
                use_pca = True
            except Exception as e:
                # Fallback if PCA fails
                transformed_samples = minority_samples.copy()
                use_pca = False
        else:
            transformed_samples = minority_samples.copy()
            
        # Scale the data
        try:
            scaled_samples = self.scaler.fit_transform(transformed_samples)
        except Exception:
            # Fallback if scaling fails
            scaled_samples = transformed_samples
            
        # Ensure we have enough minority samples to find neighbors
        n_neighbors = min(self.n_neighbors, len(scaled_samples) - 1)
        if n_neighbors <= 0:
            n_neighbors = 1
            
        synthetic_samples = []
        max_attempts = num_samples * 2  # Limit attempts to avoid infinite loops
        attempts = 0
        
        while len(synthetic_samples) < num_samples and attempts < max_attempts:
            attempts += 1
            
            # Select a random sample
            i = self._random_state.randint(0, len(scaled_samples))
            sample = scaled_samples[i]
            
            try:
                # Find k nearest neighbors
                neighbors_indices = self._find_neighbors(sample, scaled_samples, n_neighbors)
                
                # Select one of the neighbors randomly
                if len(neighbors_indices) > 0:
                    nn_idx = self._random_state.choice(neighbors_indices)
                    neighbor = scaled_samples[nn_idx]
                    
                    # Generate new sample with weighted interpolation
                    alpha = self._random_state.uniform(0, self.strategy_param)
                    new_sample = sample + alpha * (neighbor - sample)
                    
                    # Add controlled noise to increase diversity
                    noise = self._random_state.normal(0, self.noise_level, size=new_sample.shape)
                    noise = noise * self.feature_weights  # Apply feature weights to noise
                    new_sample = new_sample + noise
                    
                    synthetic_samples.append(new_sample)
            except Exception:
                # Skip this iteration if an error occurs
                continue
                
        # Handle case where we couldn't generate enough samples
        if len(synthetic_samples) == 0:
            # Fallback: Just return copies of original samples with noise
            for _ in range(num_samples):
                idx = self._random_state.randint(0, len(minority_samples))
                new_sample = minority_samples[idx].copy()
                new_sample += self._random_state.normal(0, self.noise_level, size=new_sample.shape)
                synthetic_samples.append(new_sample)
        
        synthetic_array = np.array(synthetic_samples)
        
        # Inverse transform if PCA was applied
        try:
            if use_pca and self.pca_model is not None:
                synthetic_array = self.scaler.inverse_transform(synthetic_array)
                synthetic_array = self.pca_model.inverse_transform(synthetic_array)
            else:
                synthetic_array = self.scaler.inverse_transform(synthetic_array)
        except Exception:
            # If inverse transform fails, return the untransformed samples
            pass
            
        return synthetic_array
    
    def validator(self, real_data, synthetic_data, real_labels):
        """Validate synthetic data quality with cross-validation and multiple metrics."""
        # Handle empty inputs
        if len(synthetic_data) == 0 or len(real_data) == 0:
            return 0.5, 0.0, 0.0
            
        try:
            X_val = np.vstack([real_data, synthetic_data])
            y_val = np.array([0] * len(real_data) + [1] * len(synthetic_data))
            
            # Use simplified validation for speed if data is large
            if len(X_val) > 5000:
                # Simple train-test split for large datasets
                train_size = min(int(len(X_val) * 0.7), 3000)
                indices = self._random_state.permutation(len(X_val))
                train_idx, test_idx = indices[:train_size], indices[train_size:train_size + 1000]
                
                X_train, X_test = X_val[train_idx], X_val[test_idx]
                y_train, y_test = y_val[train_idx], y_val[test_idx]
                
                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Use faster logistic regression
                clf = LogisticRegression(max_iter=200, solver='liblinear', random_state=self._random_state.randint(0, 10000))
                clf.fit(X_train_scaled, y_train)
                
                # Calculate detection rate
                y_pred = clf.predict(X_test_scaled)
                accuracy = np.mean(y_pred == y_test)
                detection_rate = 1 - accuracy
                
                # Calculate f1 score on real data classification
                real_indices = np.where(y_test == 0)[0]
                if len(real_indices) > 0:
                    real_test = X_test_scaled[real_indices]
                    real_preds = clf.predict(real_test)
                    f1_val = f1_score(np.zeros(len(real_test)), real_preds, average='binary', zero_division=0)
                else:
                    f1_val = 0.0
            else:
                # Use cross-validation for smaller datasets
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self._random_state.randint(0, 10000))
                detection_rates = []
                f1_scores = []
                
                for train_idx, test_idx in cv.split(X_val, y_val):
                    X_train, X_test = X_val[train_idx], X_val[test_idx]
                    y_train, y_test = y_val[train_idx], y_val[test_idx]
                    
                    # Scale the data
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Use logistic regression for speed
                    clf = LogisticRegression(max_iter=200, solver='liblinear', random_state=self._random_state.randint(0, 10000))
                    clf.fit(X_train_scaled, y_train)
                    
                    # Calculate detection rate
                    y_pred = clf.predict(X_test_scaled)
                    accuracy = np.mean(y_pred == y_test)
                    detection_rates.append(1 - accuracy)
                    
                    # Calculate f1 score on real data classification
                    real_indices = np.where(y_test == 0)[0]
                    if len(real_indices) > 0:
                        real_test = X_test_scaled[real_indices]
                        real_preds = clf.predict(real_test)
                        f1 = f1_score(np.zeros(len(real_test)), real_preds, average='binary', zero_division=0)
                        f1_scores.append(f1)
                
                detection_rate = np.mean(detection_rates)
                f1_val = np.mean(f1_scores) if f1_scores else 0.0
            
        except Exception as e:
            # Default values if validation fails
            detection_rate = 0.5
            f1_val = 0.0
            
        # Calculate diversity metric (simplified for speed)
        diversity = self._calculate_diversity(synthetic_data)
            
        return detection_rate, f1_val, diversity
    
    def _calculate_diversity(self, synthetic_data):
        """Calculate diversity of synthetic samples."""
        if len(synthetic_data) < 2:
            return 0.0
            
        # Calculate average pairwise distance (sampling for efficiency)
        n_samples = min(100, len(synthetic_data))  # Reduced sample size
        indices = self._random_state.choice(len(synthetic_data), n_samples, replace=False)
        samples = synthetic_data[indices]
        
        # Calculate std deviation across dimensions as a faster diversity proxy
        diversity = np.mean(np.std(samples, axis=0))
        return diversity
    
    def payoff_generator(self, detection_rate, f1_score_from_validator, diversity):
        """Enhanced payoff function considering detection rate, f1 score, and diversity."""
        # We want high detection rate (hard to distinguish), low f1 (less harm to real data),
        # and high diversity among synthetic samples
        return detection_rate - self.beta_f1_weight * f1_score_from_validator + 0.2 * diversity
    
    def _quality_filter(self, synthetic_data, real_data, threshold=0.7):
        """Filter out low-quality synthetic samples."""
        if len(synthetic_data) == 0 or len(real_data) == 0:
            return np.array([])
            
        # Simplified quality filtering for efficiency
        # Sample a subset of real data points for comparison
        n_real_samples = min(200, len(real_data))
        real_samples = real_data[self._random_state.choice(len(real_data), n_real_samples, replace=False)]
        
        # Sample synthetic data if there are too many
        if len(synthetic_data) > 500:
            indices = self._random_state.choice(len(synthetic_data), 500, replace=False)
            synthetic_subset = synthetic_data[indices]
        else:
            synthetic_subset = synthetic_data
            
        # Calculate quality scores based on distance to nearest real sample
        quality_scores = []
        for sample in synthetic_subset:
            # Calculate minimum distance to real samples
            min_distance = float('inf')
            for real_sample in real_samples:
                dist = np.sum((sample - real_sample) ** 2)  # Squared Euclidean for speed
                if dist < min_distance:
                    min_distance = dist
            
            quality = np.exp(-np.sqrt(min_distance))  # Convert back to Euclidean
            quality_scores.append(quality)
            
        # Keep samples with quality above threshold
        quality_threshold = np.percentile(quality_scores, (1-threshold)*100)
        keep_ratio = sum(q >= quality_threshold for q in quality_scores) / len(quality_scores)
        
        # If we're sampling, extrapolate the results
        if len(synthetic_subset) < len(synthetic_data):
            # Randomly select the appropriate proportion of samples
            n_keep = int(keep_ratio * len(synthetic_data))
            if n_keep > 0:
                keep_indices = self._random_state.choice(len(synthetic_data), n_keep, replace=False)
                return synthetic_data[keep_indices]
            else:
                return np.array([])
        else:
            # Use the actual quality scores
            selected_indices = np.where(np.array(quality_scores) >= quality_threshold)[0]
            return synthetic_data[selected_indices]
    
    def sample(self, X, y):
        """Generate synthetic samples for the minority class with adaptive optimization."""
        # Input validation
        if len(X) == 0 or len(y) == 0:
            return X, y
            
        class_labels, counts = np.unique(y, return_counts=True)
        
        # Handle single class case
        if len(class_labels) < 2:
            return X, y
            
        minority_class = class_labels[np.argmin(counts)]
        majority_class = class_labels[np.argmax(counts)]
        minority_samples = X[y == minority_class]
        
        # Handle empty minority class
        if len(minority_samples) == 0:
            return X, y
            
        majority_count = counts[np.argmax(counts)]
        minority_count = counts[np.argmin(counts)]
        
        # Determine number of synthetic samples to generate
        num_synthetic = majority_count - minority_count
        if num_synthetic <= 0:
            return X, y  # Classes already balanced
        
        # Calculate feature importance for weighting
        if self.feature_weights is None:
            try:
                # Use logistic regression for feature importance
                clf = LogisticRegression(penalty='l1', solver='liblinear', random_state=self._random_state.randint(0, 10000))
                clf.fit(X, y)
                # Take absolute values as importance
                self.feature_weights = np.abs(clf.coef_[0])
                # Normalize
                self.feature_weights = self.feature_weights / np.sum(self.feature_weights)
            except:
                self.feature_weights = np.ones(X.shape[1])
        
        best_synthetic_data = None
        best_payoff = -np.inf
        best_strategy_param = self.strategy_param
        
        # Adaptive parameter optimization with limited iterations
        lr = self.adaptive_lr
        patience = 3
        no_improvement = 0
        
        for iteration in range(self.max_iterations):
            # Generate synthetic samples
            synthetic_data = self.generator(minority_samples, num_samples=num_synthetic)
            
            # Validate quality
            detection_rate, f1_val, diversity = self.validator(X, synthetic_data, y)
            
            # Calculate payoff
            payoff = self.payoff_generator(detection_rate, f1_val, diversity)
            
            # Update best result
            if payoff > best_payoff:
                best_payoff = payoff
                best_synthetic_data = synthetic_data.copy()
                best_strategy_param = self.strategy_param
                no_improvement = 0
            else:
                no_improvement += 1
                
            # Adapt learning rate if no improvement
            if no_improvement >= patience:
                lr *= 0.5
                no_improvement = 0
                
                # Break early if learning rate becomes too small
                if lr < 0.001:
                    break
            
            # Update strategy parameter with adaptive learning rate
            gradient = np.sign(payoff)  # Direction of improvement
            delta = lr * gradient
            self.strategy_param += delta
            self.strategy_param = np.clip(self.strategy_param, 0.05, 1.0)
            
        # Restore best strategy parameter
        self.strategy_param = best_strategy_param
        
        # Use best synthetic data or generate new data if none found
        if best_synthetic_data is None or len(best_synthetic_data) == 0:
            best_synthetic_data = self.generator(minority_samples, num_samples=num_synthetic)
            
        # Apply quality filter to final selection (quickly)
        filtered_synthetic = self._quality_filter(best_synthetic_data, minority_samples, threshold=0.6)
            
        # If filtering removed too many samples, generate more
        if len(filtered_synthetic) < num_synthetic * 0.5:
            additional = num_synthetic - len(filtered_synthetic)
            more_synthetic = self.generator(minority_samples, num_samples=additional)
            if len(more_synthetic) > 0:
                filtered_synthetic = np.vstack((filtered_synthetic, more_synthetic))
            
        # Handle case where we have no synthetic samples
        if len(filtered_synthetic) == 0:
            return X, y
            
        y_synthetic = np.full(len(filtered_synthetic), minority_class)
        y_samp = np.append(y, y_synthetic)
        X_samp = np.vstack((X, filtered_synthetic))
        
        # Remove any invalid values
        mask = np.isfinite(X_samp).all(axis=1)
        X_samp = X_samp[mask]
        y_samp = y_samp[mask]
        
        return X_samp, y_samp
    
    def fit_resample(self, X, y):
        """Interface method for sklearn compatibility."""
        return self.sample(X, y)
    
    def get_params(self, deep=True):
        """Get parameters for this estimator."""
        return {
            'strategy_param': self.strategy_param,
            'lambda_penalty': self.lambda_penalty,
            'beta_f1_weight': self.beta_f1_weight,
            'noise_level': self.noise_level,
            'n_neighbors': self.n_neighbors,
            'feature_weights': self.feature_weights,
            'max_iterations': self.max_iterations,
            'adaptive_lr': self.adaptive_lr,
            'pca_components': self.pca_components,
            'random_state': self.random_state
        }
    
    def set_params(self, **params):
        """Set the parameters of this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        self._random_state = check_random_state(self.random_state)
        return self
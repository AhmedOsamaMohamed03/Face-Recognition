# Pattern Recognition: Face Recognition

## Team

| Name              | ID        |
|-------------------|-----------|
| Ahmed Ashraf      | 21010040  |
| Ahmed Osama       | 21010037  |
| Saifullah Mousaad | 21010651  |

## Data Manipulation

After reading the dataset, a flattening operation is performed to produce a vector containing 10,304 entries, each representing a pixel in the image.

## Principal Components Analysis (PCA)

### Approach

The steps to construct the PCA reduced space are as follows:

- Build the covariance matrix for the training set.
- Calculate the eigenvalues and eigenvectors for the covariance matrix.
- Based on the alpha value (variance to be retained), save the eigenvectors whose eigenvalues preserve the required variance as the new principal components for the new space.
- Construct matrix \( U \) representing the principal components (10,304, k_components), compute the dot product with the training data matrix with the mean subtracted (denoted as "the empirical mean") (samples, 10,304) \(\Rightarrow\) return projection of the data matrix (samples, k_components).
- Provide functions to project any matrix to the reduced space and to reconstruct it into its original space with the addition of the empirical mean.

### Results

The following table represents the number of principal components for each alpha:

| Alpha | Principal Components | Quality |
|-------|----------------------|---------|
| 0.80  | 36                   | - Training set: Some details of the faces are unrecognizable<br>- Test set: Faces are very bad |
| 0.85  | 52                   | - Training: Some faces became more recognizable<br>- Test: Still bad |
| 0.90  | 76                   | - Training: Faces are now somewhat accepted but not yet that good |
| 0.95  | 117                  | - Training: Most images became reasonable and look like the original |
| 0.99  | 174                  | - Training: Almost all faces are now the same as the original<br>- Test: Faces are still noisy, the difference between test and training sets is huge |

### Analysis

- PCA provides a mediocre-quality technique for space reduction to save memory and operating time on the dataset.
- As expected, the idea is a compromise: PCA saves a huge amount of memory and time with some losses that can be too severe for the required operations, making it less preferable.
- With increasing alpha, the amount of variance preserved increases \(\Rightarrow\) less loss and better images, but this comes with more data to be saved.
- Centering the data is crucial to ensure obtaining the highest-variance components, and adding the mean ensures images preserve good quality compared to the original.
- The training samples are reconstructed almost perfectly, while the test samples are reconstructed poorly, as the components and mean stored are for the training set.

## K-Means Clustering

### Approach

- Hard K-Means was implemented.
- Initial centroids were selected randomly from the dataset.
- Loop until convergence or reaching max iterations.
- Assign points to clusters based on Sum Square Error (SSE).
- Calculate new centroids by evaluating cluster mean values.
- Handle empty clusters by:
  - Choosing the point that contributes most to SSE.
  - Choosing a point from the cluster with the highest SSE.
- Return clusters, centroids, and cluster point indices.

### Results

#### With PCA

| Accuracy | K=20  | K=40  | K=60  |
|----------|-------|-------|-------|
| Alpha=0.8  | 0.44  | 0.67  | 0.8050 |
| Alpha=0.85 | 0.3750 | 0.5950 | 0.7250 |
| Alpha=0.9  | 0.44  | 0.6450 | 0.7900 |
| Alpha=0.95 | 0.4550 | 0.6250 | 0.78   |
| Alpha=0.99 | 0.3950 | 0.6650 | 0.79   |

#### With Autoencoder

- Accuracy of training set: 0.7300
- Test Accuracy: 0.6550

### Analysis

#### Effect of Alpha and K on K-Means Accuracy

##### Effect of Alpha on Clustering Accuracy

As alpha increases (e.g., from 0.800 to 0.975), PCA retains more principal components, increasing the dimensionality of the reduced space. K-Means relies on Euclidean distances, which become less meaningful in higher dimensions due to the "curse of dimensionality." In high-dimensional spaces, distances between points tend to become more uniform, making it harder for K-Means to distinguish meaningful clusters.

##### Effect of K on Clustering Accuracy

Increasing the number of clusters increases clustering accuracy and reduces the probability of clustering points incorrectly.

## Gaussian Mixture Model (GMM)

### Approach

- GMM was implemented using Expectation-Maximization (EM).
- Initial means were selected randomly from the dataset.
- Initialized covariances as scaled identity matrices and weights equally.
- Loop until convergence or reaching max iterations:
  - Assign points to clusters based on probabilistic responsibilities (E-step).
  - Update means, covariances, and weights using weighted means (M-step).
- Convergence checked using log-likelihood improvement.
- Worked in log space to handle numerical instability.

### Results

#### With PCA

| Accuracy | K=20  | K=40  | K=60  |
|----------|-------|-------|-------|
| Alpha=0.8  | 0.345 | 0.575 | 0.73  |
| Alpha=0.85 | 0.275 | 0.565 | 0.71  |
| Alpha=0.9  | 0.27  | 0.44  | 0.565 |
| Alpha=0.95 | 0.225 | 0.32  | 0.53  |

#### With Autoencoder

- Accuracy of training set: 0.73
- Test Accuracy: 0.65

- Test Accuracy: 0.75
- Test F1-Score: 0.7164

### Analysis

#### Accuracy vs K for Different \(\alpha\) Values

For all \(\alpha\) values, clustering accuracy improves as \( K \) increases. All lines trend upward as \( K \) increases, showing that adding more clusters improves accuracy. As \( K \) increases, the model has more components, so it can better capture different patterns or groups in the data, leading to better clustering and higher accuracy.

#### Accuracy vs \(\alpha\) for Different \( K \) Values

For all \( K \) values, accuracy decreases as \(\alpha\) increases. All lines show a downward trend as \(\alpha\) increases, meaning keeping more variance in PCA doesn't help the GMM cluster better in this case. A higher \(\alpha\) means keeping more variance, so the data has more dimensions. This might include noise or less important details that make it harder for the GMM to find clear clusters, lowering accuracy.

GMM models each cluster as a Gaussian distribution, which involves estimating means and covariances. In higher dimensions (higher \(\alpha\)), the covariance matrices become harder to estimate accurately, especially if the data points are spread out or noisy. This can lead to overlapping clusters, where points are assigned to the wrong group, reducing accuracy.

## GMM vs K-Means

### Comparison on Test Set

#### On PCA

| Model   | Accuracy | F1-Score |
|---------|----------|----------|
| GMM     | 0.75     | 0.7164   |
| K-Means | 0.715    | 0.7106   |

#### On Autoencoder

| Model   | Accuracy | F1-Score |
|---------|----------|----------|
| GMM     | 0.64     | 0.7414   |
| K-Means | 0.665    | 0.6159   |

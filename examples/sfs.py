import numpy as np
import physbo
from physbo.misc import SetConfig
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
import matplotlib.pyplot as plt


# 1. Generate data
def generate_data(n_samples=2000, n_features=5):
    """Generate artificial dataset"""
    # Generate features
    X = np.random.uniform(-10, 10, (n_samples, n_features))

    # Define objective function
    def target_function(x):
        # Function where x1, x2, x5 are important features
        return (
            (100 - x[4]) ** 2
            + (10 - x[0]) ** 2
            + (1 - x[1]) ** 2
            + x[3]
            + (1000 - x[0] * x[1])
            + np.random.normal(0, 1)
        )

    # Calculate target variable
    y = np.array([target_function(x) for x in X])

    return X, y


# 2. GP model setup
def setup_gp_model(input_dim):
    """Setup GP model"""
    # Set kernel (covariance function)
    cov = physbo.gp.cov.gauss(input_dim, ard=False)

    # Set mean function
    mean = physbo.gp.mean.const()

    # Set likelihood function
    lik = physbo.gp.lik.gauss()

    # Prepare configuration
    config = SetConfig()

    # Create GP model
    gp = physbo.gp.sfs(lik=lik, mean=mean, cov=cov, config=config)

    return gp


def main():
    # 1. Generate data
    X, y = generate_data()
    print("Dataset shape:", X.shape)

    # 2. Setup GP model
    gp = setup_gp_model(X.shape[1])

    # 3. Configure Sequential Feature Selector
    sfs = SFS(
        estimator=gp,
        k_features=3,  # Number of features to select
        forward=True,  # Use forward selection
        floating=True,  # Use Floating search
        scoring="r2",  # Evaluation metric
        cv=3,  # Number of cross-validation splits
        n_jobs=-1,  # Number of parallel jobs
    )

    # 4. Execute feature selection
    sfs.fit(X, y)

    # 5. Display results
    print("\nSelected features:")
    print(sfs.subsets_)

    # 6. Visualize results
    plt.figure(figsize=(10, 6))
    from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

    fig = plot_sfs(sfs.get_metric_dict(), kind="std_err")
    plt.title("Sequential Forward Selection (w/ SFFS)")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()

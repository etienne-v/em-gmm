
# generate random data in clusters
def generateRandomClusterData(nPoints, weights, means, cov):
    """
    Generate nPoints random data points from a Gaussian distribution and in different
    clusters as specified by parameters.
    """
    import numpy as np
    assert len(weights) == len(means)
    assert len(weights) == len(cov)
    nClusters = len(weights)
    data = []
    for i in np.arange(nPoints):
        clusterNum = np.random.choice(nClusters, size=1, p=weights)[0]
        data.append(np.random.multivariate_normal(mean=means[clusterNum], cov=cov[clusterNum], size=1)[0])
    return np.array(data)


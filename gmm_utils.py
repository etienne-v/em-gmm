

def generateRandomClusterData(nPoints, weights, means, covs):
    """
    Generate nPoints random data points from a Gaussian distribution and in different
    clusters as specified by parameters.
    """
    import numpy as np
    assert len(weights) == len(means)
    assert len(weights) == len(covs)
    nClusters = len(weights)
    data = []
    for i in np.arange(nPoints):
        clusterNum = np.random.choice(nClusters, size=1, p=weights)[0]
        data.append(np.random.multivariate_normal(mean=means[clusterNum], cov=covs[clusterNum], size=1)[0])
    return np.array(data)


def calcResponsibilities(data, params):
    """
    Calculate cluster responsibilities for each data point for the given cluster parameters.
    """
    import numpy as np
    from scipy.stats import multivariate_normal

    weights = params['weights']
    means = params['means']
    covs = params['covs']

    nData = len(data)
    nClusters = len(weights)
    resp = np.zeros((nData, nClusters))
    for i in np.arange(nData):
        normConst = 0
        for k in np.arange(nClusters):
            resp[i,k] = weights[k]*multivariate_normal.pdf(data[i], mean=means[k], cov=covs[k])
        resp[i,:] = resp[i,:]/resp[i,:].sum()

    return np.array(resp)


# plot data and probability contours for clusters
def plotDataContours(data, params, trueParams, figTitle='', save=False, fileName='figure'):
    """
    Plot data points and overlay contour lines of estimated clusters for the given cluster parameters.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.mlab as mlab

    weights = params['weights']
    means = params['means']
    covs = params['covs']
    
    nClusters = len(means)
    x = np.arange(min(data[:,0]), max(data[:,0]), 0.1)
    y = np.arange(min(data[:,1]), max(data[:,1]), 0.1)
    X, Y = np.meshgrid(x, y)

    colAr = ['red', 'blue']
    
    fig = plt.figure(figsize=(10,4))
    ax1 = plt.subplot(1,2,1)
    
    plt.plot(data[:,0], data[:,1], color='0.5', marker='o', ls='None', mec='k')
    plt.xlabel("X")
    plt.ylabel("Y")

    for k in np.arange(nClusters):
        sigx = np.sqrt(covs[k][0][0])
        sigy = np.sqrt(covs[k][1][1])
        sigxy = covs[k][0][1]/(sigx*sigy)
        Z = mlab.bivariate_normal(X, Y, sigmax=sigx, sigmay=sigy, mux=means[k,0], muy=means[k,1], sigmaxy=sigxy)
        plt.contour(X, Y, Z, zorder=10, colors=colAr[k])
        plt.title(figTitle)
        plt.axis('scaled')
        plt.tight_layout()

    ax2 = plt.subplot(1,2,2)
    plt.axis('off')

    d = 0.055
    skip = 0.4
    for k in np.arange(nClusters):
        vpos = 1.0-skip*k
        plt.text(0.0, vpos-d*0, "Cluster {:d}".format(k), transform=ax2.transAxes, va='top')
        plt.text(0.0, vpos-d*1, "true weight:\n"+np.array_str(trueParams['weights'][k], precision=3, suppress_small=True), transform=ax2.transAxes, va='top')
        plt.text(0.33, vpos-d*1, "true means:\n"+np.array_str(trueParams['means'][k], precision=3, suppress_small=True), transform=ax2.transAxes, va='top')
        plt.text(0.66, vpos-d*1, "true cov:\n", transform=ax2.transAxes, va='top')
        plt.text(0.66, vpos-d*2, np.array_str(trueParams['covs'][k], precision=3, suppress_small=True), transform=ax2.transAxes, va='top')

        plt.text(0.0, vpos-d*4, "current weight:\n"+np.array_str(np.round(weights[k], 3), precision=3, suppress_small=True), transform=ax2.transAxes, va='top', color=colAr[k])
        plt.text(0.33, vpos-d*4, "current means:\n"+np.array_str(means[k], precision=3, suppress_small=True), transform=ax2.transAxes, va='top', color=colAr[k])
        plt.text(0.66, vpos-d*4, "current cov:\n", transform=ax2.transAxes, va='top', color=colAr[k])
        plt.text(0.66, vpos-d*5, np.array_str(covs[k], precision=3, suppress_small=True), transform=ax2.transAxes, va='top', color=colAr[k])

    
    if save:
        plt.savefig(fileName+'.pdf', format='pdf', bbox_inches='tight')

    plt.show()
    

def maximizeLikelihood(data, resp):
    """
    Maximize the likelihood over the cluster parameters given the current responsibilities.
    """
    import numpy as np

    nClusters = len(resp[0,:])
    nData = len(data[:,0])
    nDim = len(resp[0,:])

    # calculate weights
    nkSoft = np.sum(resp, axis=0)
    nDataResp = np.sum(np.sum(resp, axis=0))
    piHat = np.array(nkSoft/nDataResp)

    # calculate means
    muHat = []
    for k in np.arange(nClusters):
        summation = 0
        for i in np.arange(nData):
            summation += resp[i,k]*data[i]
        muHat.append( 1.0/nkSoft[k]*summation )
    muHat = np.array(muHat)

    # calculate covariances
    sigmaHat = []
    for k in np.arange(nClusters):
        summation = 0
        for i in np.arange(nData):
            summation += resp[i,k]*np.outer(data[i]-muHat[k], (data[i]-muHat[k]).T)
        sigmaHat.append( 1.0/nkSoft[k]*summation )
    sigmaHat = np.array(sigmaHat)
    
    parameters = {'weights':piHat, 'means':muHat, 'covs':sigmaHat}
    
    return parameters


def gmm_em(data, params, trueParams, maxSteps=20):
    """
    Run expectation maximization algorithm for fitting a Gaussian mixture model to clusters of data.
    """
    import numpy as np
    import time
    from gmm_utils import plotDataContours, calcResponsibilities, maximizeLikelihood

    # initialize parameters
    weights = params['weights']
    means = params['means']
    covs = params['covs']
    resp = []
    
    for step in np.arange(maxSteps):

        # plot data and contour lines with current cluster parameters
        plotDataContours(data, params, trueParams, figTitle="Step {:d}".format(step), save=True, fileName="figures/em_{:d}".format(step))

        # calculate responsibilities with current cluster parameters
        resp = calcResponsibilities(data, params)

        # maximize likelihood over cluster parameters with current responsibilities
        params = maximizeLikelihood(data, resp)

    plotDataContours(data, params, trueParams, figTitle="Step {:d}".format(step))
    
    return params



# def calcCov(data, means):
#     """
#     Calculate diagonal covariance matrix with predefined mean values.
#     """
#     import numpy as np

#     nClusters = len(means)
#     dim = len(means[0])
#     covMats = []
#     for cluster in np.arange(nClusters):
#         cov = (np.dot((data[:,0] - means[cluster, 0]).T, data[:,1] - means[cluster, 1]).sum()) / (len(data)-1)
#         covMat = np.zeros((dim, dim))
#         np.fill_diagonal(covMat, cov)
#         covMats.append(covMat)

#     return np.array(covMats)

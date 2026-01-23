import numpy as np

threshhold = 1e-6
def threshold(array, thresh):
    return [0 if abs(a)<thresh else a for a in array]

def intersection(a: np.array, b: np.array) -> np.array:
    return np.array(list(set(a).intersection(b)))

def TPR(theta, thetahat):  #True Positive Ratio

    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)

    nonzeros_theta = np.nonzero(theta.flatten())[0]
    nonzeros_thetahat = np.nonzero(thetahat)[0]

    tpr = len(intersection(nonzeros_theta, nonzeros_thetahat)) / len(nonzeros_theta)

    return tpr


def FPR(theta, thetahat):  #True Positive Ratio

    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)

    nonzeros_theta = np.nonzero(theta.flatten())[0]
    nonzeros_thetahat = np.nonzero(thetahat)[0]

    fpr = 1 - len(intersection(nonzeros_theta, nonzeros_thetahat)) / len(nonzeros_theta)

    return fpr

def TNR(theta, thetahat):  #True Negative Ratio

    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)

    zeros_theta = np.where(theta.flatten() == 0)[0]
    zeros_thetahat = np.where(thetahat == 0)[0]

    tnr = len(intersection(zeros_theta, zeros_thetahat)) / len(zeros_theta)

    return tnr

def FNR(theta, thetahat):  #False Negative Ratio    

    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)

    nonzeros_theta = np.nonzero(theta.flatten())[0]
    nonzeros_thetahat = np.nonzero(thetahat)[0]

    fnr = 1 - len(intersection(nonzeros_theta, nonzeros_thetahat)) / len(nonzeros_theta)

    return fnr

def MCC(theta, thetahat):  #Matthews Correlation Coefficient
    if len(theta) != len(thetahat):
        raise ValueError('Not the same lengths')
    thetahat = threshold(thetahat, threshhold)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(theta)):
        if theta[i] != 0 and thetahat[i] != 0:
            tp += 1
        elif theta[i] == 0 and thetahat[i] == 0:
            tn += 1
        elif theta[i] == 0 and thetahat[i] != 0:
            fp += 1
        elif theta[i] != 0 and thetahat[i] == 0:
            fn += 1
    numerator = (tp * tn) - (fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return 0
    else:
        return numerator / denominator
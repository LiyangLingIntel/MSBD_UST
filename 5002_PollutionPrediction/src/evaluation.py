import numpy as np

def smape(actual, predicted):
    dividend= np.abs(np.array(actual) - np.array(predicted))
    c = np.array(actual) + np.array(predicted)
    
    return 2 * np.mean(np.divide(dividend, c, out=np.zeros_like(dividend), where=c!=0, casting='unsafe'))
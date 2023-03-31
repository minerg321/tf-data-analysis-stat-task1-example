import pandas as pd
import numpy as np
from scipy.optimize import minimize

chat_id = 575072396 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array) -> float:
   # Определение функции правдоподобия модели
    def likelihood_func(mu: float, x: np.array) -> float:
        n = len(x)
        return -n * np.log(mu) - np.sum(x) / mu

    # Максимизация функции правдоподобия
    res = minimize(likelihood_func, x0=1, args=(x,), method='Nelder-Mead')
    
    # Возвращение оценки параметра модели
    return res.x[0]

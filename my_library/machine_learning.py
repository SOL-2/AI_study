import numpy as np



def numerical_derivative(f,x):
    
    # f : 미분하려고 하는 다변수 함수
    # x : 모든 변수를 포함하고 있어야 한다. ndarray(차원 상관없이)
    
    delta_x = 1e-4
    derivative_x = np.zeros_like(x) # 미분한 결과를 저장하는 ndarray
    
    # iterator를 이용해서 입력변수 x에 대해 편미분을 수행
    
    it = np.nditer(x, flags=['multi_index'])
    
    while not it.finished:
        
        # iterator의 현재 index를 추출(tuple로 추출)
        idx = it.multi_index  
        
        # 현재 칸의 값을 어딘가에 잠시 저장
        tmp = x[idx]
        
        x[idx] = tmp + delta_x
        fx_plus_delta = f(x)  # f(x + delta_x)
        
        
        x[idx] = tmp - delta_x
        fx_minus_delta = f(x) # f(x - delta_x)
        
        derivative_x[idx] = (fx_plus_delta - fx_minus_delta) / (2 * delta_x)
        
        x[idx] = tmp # 데이터를 원상 복구
        
        it.iternext()
        
    
    return derivative_x


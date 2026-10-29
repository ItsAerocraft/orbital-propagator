import numpy as np
from control.jacobian import return_test_derivatives, jacobian

def test_jacobian_finite_difference():
    I = np.diag([2.0, 3.0, 4.0])
    I_inv = np.linalg.inv(I)
    
    np.random.seed(30) 

    h = 1e-6 #step for finite difference   
    tol = 1e-4 #tolerance bound for result
    
    for i in range(50):

        #random q_vec and omegas
        q_vec = np.random.randn(3) * 0.1  
        omega = np.random.randn(3) * 0.5
        
        x = np.concatenate([q_vec, omega])

        A_test = jacobian(x, I, I_inv)
        
        A_numerical = np.zeros((6, 6))

        for j in range(6):
            e_j = np.zeros(6)
            e_j[j] = h
            
            f_plus = return_test_derivatives(x + e_j, I, I_inv)
            f_minus = return_test_derivatives(x - e_j, I, I_inv)
            
            # Central difference --> (f(x+h) - f(x-h)) / 2h
            A_numerical[:, j] = (f_plus - f_minus) / (2.0 * h)
            
        # np.allclose is a function that checks if |A_analytical - A_numerical| < tol element-wise
        if not np.allclose(A_test, A_numerical, atol = tol):
            print(f"Test failed at state {i}:")
            print("x =", x)
            print("Testing:\n", A_test)
            print("Numerical:\n", A_numerical)
            print("Difference:\n", A_test - A_numerical)
            assert False, "mismatch between test and numerical Jacobians"
            
    print("Test passed: test and numerical Jacobians match.")

if __name__ == "__main__":
    test_jacobian_finite_difference()
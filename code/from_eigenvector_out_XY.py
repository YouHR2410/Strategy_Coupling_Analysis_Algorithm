import numpy as np
from sympy import *
import math
import sympy
import nashpy as nash


def from_eigenvector_out_X(v_1column):
    Ndim = len(v_1column)
    Ymn = []
    Xmn = []
    Tmn = np.zeros((Ndim, Ndim))
    Smn = np.zeros((Ndim, Ndim))

    for m in range(Ndim-1):
        for n in range(m+1, Ndim):
            vm = v_1column[m]
            vn = v_1column[n]

            area_m_n = abs(vm) * abs(vn) * np.pi * math.sin(arg(vm).evalf() - arg(vn).evalf())  # Eq 1
            Ymn.append(area_m_n.evalf())
            Tmn[m, n] = area_m_n
            Tmn[n, m] = -area_m_n

    for m in range(Ndim):
        for n in range(m, Ndim):
            vm = v_1column[m]
            vn = v_1column[n]

            brea_m_n = abs(vm) * abs(vn) * np.pi * math.cos(arg(vm).evalf() - arg(vn).evalf())  # Eq 1
            Xmn.append((float)(brea_m_n.evalf()))
            Smn[m, n] = brea_m_n
            Smn[n, m] = brea_m_n

    return Xmn


#mode_col can be set 3 or 4
def A5_matrix_X(a_value,mode_col):
    x1, x2, x3, x4, x5,a= symbols('x1 x2 x3 x4 x5 a', real=True)
    payoff_matrix = Matrix([[0, a, 1, -1, -a],
                            [-a, 0, a, 1, -1],
                            [-1, -a, 0, a, 1],
                            [1, -1, -a, 0, a],
                            [a, 1, -1, -a, 0]])

    Payoff_vector_field_F = payoff_matrix * Matrix([x1, x2, x3, x4, x5])

    mean_U = Matrix([x1, x2, x3, x4, x5]).T * Payoff_vector_field_F

    mean_U_5 = Matrix([mean_U,mean_U,mean_U,mean_U,mean_U])

    V_F = sympy.matrices.dense.matrix_multiply_elementwise(Matrix([x1, x2, x3, x4, x5]),(Payoff_vector_field_F- mean_U_5))
    D_V_F = Matrix([diff(V_F, x1).T, diff(V_F, x2).T, diff(V_F, x3).T, diff(V_F, x4).T, diff(V_F, x5).T]).T
 
    #equilibrium is (0.2,0.2,0.2,0.2,0.2)
    eigen = D_V_F.subs([(x1,0.2),(x2,0.2),(x3,0.2),(x4,0.2),(x5,0.2),(a,a_value)]).eigenvects()


    eigenvector1 = eigen[0][2][0].normalized()
    eigenvector2 = eigen[1][2][0].normalized()
    eigenvector3 = eigen[2][2][0].normalized()
    eigenvector4 = eigen[3][2][0].normalized()
    eigenvector5 = eigen[4][2][0].normalized()

    eigenvectormatrix = Matrix([eigenvector1.T, eigenvector2.T , eigenvector3.T,eigenvector4.T,eigenvector5.T]).T

    V2 = payoff_matrix.subs([(a,a_value)]) * eigenvectormatrix
    V3 = eigenvectormatrix[:,mode_col]
    Xmn = from_eigenvector_out_X(V3)
    return Xmn


#No arguments required
def Y5_matrix_X():
    x1, x2, x3, x4, x5= symbols('x1 x2 x3 x4 x5', real=True)
    payoff_matrix = Matrix([[0, 3, 4, 11, 11],
                            [5, 0, 2, 11, 12],
                            [2, 5, 0, 9, 12],
                            [6, 10, 10, 0, 3],
                            [10, 10, 10, 4, 0]])

    Payoff_vector_field_F = payoff_matrix * Matrix([x1, x2, x3, x4, x5])
    mean_U = Matrix([x1, x2, x3, x4, x5]).T * Payoff_vector_field_F

    mean_U_5 = Matrix([mean_U,mean_U,mean_U,mean_U,mean_U])
    V_F = sympy.matrices.dense.matrix_multiply_elementwise(Matrix([x1, x2, x3, x4, x5]),(Payoff_vector_field_F- mean_U_5))


    D_V_F = Matrix([diff(V_F, x1).T, diff(V_F, x2).T, diff(V_F, x3).T, diff(V_F, x4).T, diff(V_F, x5).T]).T
   
    payoff_matrix = np.array([[0, 3, 4, 11, 11],
                            [5, 0, 2, 11, 12],
                            [2, 5, 0, 9, 12],
                            [6, 10, 10, 0, 3],
                            [10, 10, 10, 4, 0]])

    game = nash.Game(payoff_matrix)
    eqs = game.support_enumeration()
    t = list(eqs)

    # t = [(array([0.05992635, 0.28054905, 0.12420489, 0.21258788, 0.32273184]),
    #       array([0.14864412, 0.21359223, 0.21459659, 0.09641781, 0.32674925]))]
    values = {x1:t[0][1][0],x2:t[0][1][1],x3:t[0][1][2],x4:t[0][1][3],x5:t[0][1][4]}
    eigen = D_V_F.evalf(subs=values, n=4).eigenvects()

    eigenvector1 = eigen[0][2][0].normalized()
    eigenvector2 = eigen[1][2][0].normalized()
    eigenvector3 = eigen[2][2][0].normalized()
    eigenvector4 = eigen[3][2][0].normalized()
    eigenvector5 = eigen[4][2][0].normalized()

    eigenvectormatrix = Matrix([eigenvector1.T, eigenvector2.T , eigenvector3.T,eigenvector4.T,eigenvector5.T]).T

    V3 = eigenvectormatrix[:,2]
    Xmn = from_eigenvector_out_X(V3)
    return Xmn




def A4_matrix_X(a_value):
    x1, x2, x3, x4,a= symbols('x1 x2 x3 x4 a', real=True)
    payoff_matrix = Matrix([[0, 0, 0, a], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])

    Payoff_vector_field_F = payoff_matrix * Matrix([x1, x2, x3, x4])
    mean_U = Matrix([x1, x2, x3, x4]).T * Payoff_vector_field_F
    mean_U_5 = Matrix([mean_U,mean_U,mean_U,mean_U])
  
    V_F = sympy.matrices.dense.matrix_multiply_elementwise(Matrix([x1, x2, x3, x4]),(Payoff_vector_field_F- mean_U_5))
    D_V_F = Matrix([diff(V_F, x1).T, diff(V_F, x2).T, diff(V_F, x3).T, diff(V_F, x4).T]).T
    S = solve((V_F),(x1,x2,x3,x4))
  

    #S = [(0, x2, 0, x4),
    #     (x1, 0, 0, 0),
    #     (x1, 0, x3, 0),
    #     (a/(3*a + 1), a/(3*a + 1), a/(3*a + 1), 1/(3*a + 1))]
    eigen = D_V_F.subs([(x1,S[3][0]),(x2,S[3][1]),(x3,S[3][2]),(x4,S[3][3])]).eigenvects()

    eigenvector1 = eigen[0][2][0].normalized()
    eigenvector2 = eigen[0][2][1].normalized()
    eigenvector3 = eigen[1][2][0].normalized()
    eigenvector4 = eigen[2][2][0].normalized()

    eigenvectormatrix = Matrix([eigenvector1.T, eigenvector2.T , eigenvector3.T,eigenvector4.T]).T

    V3 = eigenvectormatrix[:,2]
    Xmn = from_eigenvector_out_X(V3.subs([(a,a_value)]))
    return Xmn

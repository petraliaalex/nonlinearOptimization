"""
Alex Petralia
M/CS 435
Assignment 1: Line search methods (Corrected Version)
"""
from sympy import *
import numpy as np
from numpy.linalg import inv
import pdb #debugger

#Armijo function
def Armijo(xk,fcn,pk,gradk,c,r):
    iter_conv = 0 #numEvals = 0
    alpha = 1    #our alpha value
    maxIter = 30 #threshold for max iterations possible
    curIter = 0 #current number of iterations done
    finalIterCount = 0 #final number of iterations done
    convValue = [0,0] #convergence value
    while curIter <= maxIter:
      LHS = fcn(xk + alpha * pk)[0]
      RHS = fcn(xk)[0] + alpha * c * gradk * pk  
      if LHS <= RHS:
         finalIterCount = curIter
         convValue = xk + alpha * pk #equivalent as: xk += alpha * pk 
         break                                             
      else:
         alpha = alpha * r
      curIter += 1
    finalconvValue = convValue
    return finalconvValue, finalIterCount
    
#Newton Line Search (set up to do one step)
def Newton2(xk,fcn,grad,hess,pk,gradk,c,ctilde,tol):
   totIter = 0 #total iterations made before ToL condition is met
   alpha = 1
   newAlpha = 2
   curIter = 0
   maxIter = 1
   #Strong Wolfe Conditions
   LHS1 = fcn(xk + alpha * pk)[0]
   RHS1 = fcn(xk)[0] + c*alpha*pk*fcn(xk)[0]
   LHS2 = abs(pk*fcn(xk + alpha * pk)[1])
   RHS2 = ctilde * abs(pk*fcn(xk)[1])
   while curIter < maxIter: #determines number of steps (currently allows 1 step)
      while abs(newAlpha - alpha) > tol: #and curIter < maxIter:
         if LHS1 > RHS1 and LHS2 > RHS2: #and curIter < maxIter:
            curIter += 1
            totIter += 1
            gamma = fcn(xk + alpha * pk)[0]
            gammaD = fcn(xk + alpha * pk)[1] * pk
            newAlpha = alpha - (gamma/gammaD)
            alpha = newAlpha
            xk = xk + alpha * pk
   return xk, curIter


#-------------------------test functions------------------
def testFunction1(x):
    fval = x**16-x**2+x-1
    fgrad = 16*x**15-2*x+1
    fHess = (16*15)*x**14-2
    return fval,fgrad,fHess
def testFunction2(x):
    fval = .01*(x**6-30*x**4+19*x**2+7*x**3)
    fgrad = .01*(6*x**5-120*x**3+38*x+21*x**2)
    fHess = .01*(30*x**4-360*x**2+38+42*x)
    return fval,fgrad,fHess
#------------------------------------------------------------


#------------------(Armijo)Initialized values/definitions----
def gradFxn(x): #grad function
   return testFunction1(x)[1] 
def hessFxn(x): #hess function
   return testFunction1(x)[2]
xk1 = 1 #xk vector
pk1 = -gradFxn(xk1) #pk vector/direction
gradk1 = testFunction1(xk1)[1] #gradk value
c1 = 10**-4 #c value in (0,1)
r1 = 0.5 #r value in (0,1)
ctilde1 = 0.7 #ctilde value where 0 < c < ctilde < 1
ToL1 = 0.01 #Tolerance value
#-----------------------------------------------------------


#------------------(Newton)Initialized values/definitions---
xk2 = 2 #xk vector
gradk2 = testFunction1(xk2)[1] #gradk value
pk2 = -gradFxn(xk2) #pk vector/direction
c2 = 10**-4 #c value in (0,1)
r2 = 0.5 #r value in (0,1)
ctilde2 = 0.7 #ctilde value where 0 < c < ctilde < 1
ToL2 = 0.01 #Tolerance value
#------------------------------------------------------------


#----------------------execution----------------------------------------
#run Armijo()
xkFinal, iterFinal = Armijo(xk1, testFunction1, pk1, gradk1, c1, r1)

#run Newton()
xkFinal2, iterFinal2 = Newton2(xk2, testFunction1, gradFxn, hessFxn, pk2, gradk2, c2, ctilde2, ToL2)
#-------------------------------------------------------------------------

#------------------------Print statements---------------------------------
print("(Armijo)Before stepping:")
print("---------------:")
print("xk = " + str(xk1))
print("numEvals = " + str(0))
print("f(xk) = " + str(testFunction1(xk1)[0]))
print("\n")
print("(Armijo)After stepping:")
print("---------------:")
print("xk = " + str(xkFinal))
print("numEvals = " + str(iterFinal))
print("f(xk) = " + str(testFunction1(xkFinal)[0]))
print("\n")
print("\n")
print("(Newton)Before stepping:")
print("---------------:")
print("xk = " + str(xk2))
print("numEvals = " + str(0))
print("f(xk) = " + str(testFunction1(xk2)[0]))
print("\n")
print("(Newton)After stepping:")
print("---------------:")
print("xk = " + str(xkFinal2))
print("numEvals = " + str(iterFinal2))
print("f(xk) = " + str(testFunction1(xkFinal2)[0]))
#---------------------------------------------------------------------------

if __name__ == "__main__":
   print("")
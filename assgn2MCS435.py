"""
Alex Petralia
M/CS 435
Assignment 2: Update Directions

Algorithms/Methods Used:
   -Steepest Descent Method
   -DFP Algorithm
"""
from sympy import *
import numpy as np
from numpy.linalg import inv
import pdb #debugger
from numpy import linalg as LA


#------------------Steepest Descent Method--------------------------------------------------------------------------
def sDescent(xk, eps, fxn):
   global fxnCalls
   alpha = 10**-4 # step size multiplier
   maxIters = 2 # maximum number of iterations
   curIters = 0 #iteration counter
   xk0 = 0 #initializes xk0
   df = lambda x: 16*x**15-2*x+1
   pk = -df(xk)  
   fxnCalls = fxnCalls + 1
   print("Steepest Descent Method")
   print("----------Initial Values------------")
   print("current iteration = " + str(curIters))
   print("current xk = " + str(xk))
   print("current f(xk) = " + str(testFunction1(xk)[0]))
   print("current ||grad(xk)|| = " + str(LA.norm(testFunction1(xk)[1])))
   print("current total function calls = " + str(fxnCalls))
   print("\n")
   while LA.norm(pk) > eps and curIters < maxIters: 
       pk = -df(xk) #pk = -grad(xk) saves the direction
       fxnCalls = fxnCalls + 1
       a, b, newAlpha, calls = Armijo(xk, testFunction1, pk, testFunction1(xk)[1], alpha, 0.5, 0.7) #search for alpha with Armijo
       fxnCalls = fxnCalls + 1
       fxnCalls += calls
       alpha = newAlpha
       xk = xk + alpha * np.dot(alpha, pk) #<=> xk+1 = xk + alpha*pk
       curIters = curIters + 1
       print("current iteration = " + str(curIters))
       print("current xk = " + str(xk))
       print("current f(xk) = " + str(testFunction1(xk)[0]))
       print("current ||grad(xk)|| = " + str(LA.norm(testFunction1(xk)[1])))
       print("current total function calls = " + str(fxnCalls))
       print("\n")
   fxnCalls = 0 #resets the total function calls after the algorithm concludes
#---------------------------------------------------------------------------------------------------------------------------------------


#------------------DFP Algorithm-----------------------------------------------------------------
def DFP(xk, eps, fxn, Hk):
   #0) set values 
   #1) compute Hki
   #2) pk = -Hk*grad(xk)
   #3) set gamma(alpha) = f(xk + alpha * pk)
   #4) line search with Armijo for -> alpha
   #5) set xk+1 = xk + alpha * pk
   
   #note: ROF = rate of change/triangle
   #6) set ROF(xk) = xk+1 -xk
   #7) set ROF(gk) = grad(xk+1) - grad(xk)
   #8) set Hk+1 = Hk + ... - ... (algorithm for Hk+1 on lecture notes)
   #9) Repeat with k = k + 1 
   
   alpha = 1   #step 0
   curIters = 0
   maxIters = 2 #max iteration threshold
   fxnCalls = 0 #keeps track of total function calls
   #pdb.set_trace()
   print("DFP ALgorithm")
   print("----------Initial Values------------")
   print("current iteration = " + str(curIters))
   print("current xk = " + str(xk))
   print("current f(xk) = " + str(testFunction1(xk)[0]))
   print("current ||grad(xk)|| = " + str(LA.norm(testFunction1(xk)[1])))
   print("current total function calls = " + str(fxnCalls))
   print("\n")
   
   while LA.norm(fxn(xk)[1]) > eps and curIters < maxIters:
      curIters += 1
      pk = np.dot(-Hk, fxn(xk)[1])#step 2
      fxnCalls += 1
      a, b, newAlpha, calls = Armijo(xk, fxn, pk, fxn(xk)[1], alpha, 0.5, 0.7)#step 3 & 4
      fxnCalls += 1
      fxnCalls += calls
      alpha = newAlpha
      xk1 = xk + np.dot(alpha, pk)#step 5
      xkD = xk1 - xk#step 6
      gkD = fxn(xk1)[1] - fxn(xk)[1]#step 7
      fxnCalls += 2
      xkRatio = np.dot(xkD, np.transpose(xkD))/np.dot(np.transpose(xkD), gkD)
      hkRatio = (np.dot(Hk, gkD)*np.transpose(np.dot(Hk, gkD))) / (np.dot(np.dot(np.transpose(gkD), Hk), gkD))
      Hk1 = Hk + xkRatio - hkRatio#step 8
      xk = xk1  #step 9
      H = Hk
      print("current iteration = " + str(curIters))
      print("current xk = " + str(xk))
      print("current f(xk) = " + str(testFunction1(xk)[0]))
      print("current ||grad(xk)|| = " + str(LA.norm(testFunction1(xk)[1])))
      print("current total function calls = " + str(fxnCalls))
      print("\n")
   fxnCalls = 0 #resets fxnCalls
#-------------------------------------------------------------------------------------------------------------------







#------------------Armijo Line Search--------------------------------------------------------------------------------------------------
def Armijo(xk,fcn,pk,gradk,c,r, ctilde):
    fxnCalls0 = 0
    iter_conv = 0 #numEvals = 0
    alpha = 1    #our alpha value
    maxIter = 30 #threshold for max iterations possible
    curIter = 0 #current number of iterations done
    finalIterCount = 0 #final number of iterations done
    convValue = [0,0] #convergence value
    while curIter <= maxIter:
      LHS1 = fcn(xk + np.dot(alpha, pk))[0]
      fxnCalls0 = fxnCalls0 + 1
      RHS1 = fcn(xk)[0] + alpha * c * np.dot(fcn(xk)[1], pk)
      fxnCalls0 += 2
      #LHS2 = np.dot(fcn(xk + np.dot(alpha, pk))[1], pk)    ##use these 3 lines if you want the 2nd armijo condition(adds more fxn calls)
      #fxnCalls = fxnCalls + 1
      #RHS2 = ctilde * np.dot(gradk, pk)
      if (LHS1 <= RHS1).all(): #& (LHS2 >= RHS2).all():     ##use 2nd boolean if using 2nd condition
         finalIterCount = curIter
         convValue = xk + c * pk #equivalent as: xk += alpha * pk 
         break                                             
      else:
         alpha = alpha * r
      curIter += 1
    finalconvValue = convValue
    return finalconvValue, finalIterCount, alpha, fxnCalls0
#-------------------------------------------------------------------------------------------------------------------------------------------

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
#------------------------------------------------------------------------


#------------------Initialized values/definitions-------------------------
def gradFxn(x): #grad function
   return testFunction1(x)[1] 
def hessFxn(x): #hess function
   return testFunction1(x)[2]
xk1 = np.array([1,0])
pk1 = -gradFxn(xk1) #pk vector/direction
gradk1 = testFunction1(xk1)[1] #gradk value
c1 = 0.1*10**-4 #c value in (0,1)
r1 = 0.5 #r value in (0,1)
ctilde1 = 0.7 #ctilde value where 0 < c < ctilde < 1
ToL = 0.01 #Tolerance value
eps = 0.00001
fxnCalls = 0 #global variable
H = np.identity(2) #intitial, pos-def matrix 
#-----------------------------------------------------------------------



#----------------------execution----------------------------------------
sDescent(xk1, eps, testFunction1)
DFP(xk1, eps, testFunction1, H)
#-----------------------------------------------------------------------





#---------------print statements for testing ------------------------------------------------------------
#run Armijo()
#xkFinal, iterFinal, alpha1, calls = Armijo(xk1, testFunction1, pk1, gradk1, c1, r1, ctilde1)
#print("(Armijo)Before stepping:")
#print("---------------:")
#print("xk = " + str(xk1))
#print("numEvals = " + str(0))
#print("f(xk) = " + str(testFunction1(xk1)[0]))
#print("\n")
#print("(Armijo)After stepping:")
#print("---------------:")
#print("xk = " + str(xkFinal))
#print("numEvals = " + str(iterFinal))
#print("f(xk) = " + str(testFunction1(xkFinal)[0]))
#print("\n")

#print("TOTAL FXN CALLS: " + str(fxnCalls))
#run sDescent()
#xkFinal2, fkFinal2, gradkFinal2, hesskFinal2, iterFinal2 = sDescent(xk1, testFunction1, eps, ToL)
#print("\n")
#print("(Steepest Descent)Before stepping:")
#print("---------------:")
#print("xk = " + str(xk1))
#print("numEvals = " + str(0))
#print("f(xk) = " + str(testFunction1(xk1)[0]))
#print("grad(xk) = " + str(testFunction1(xk1)[1]))
#print("hess(xk) = " + str(testFunction1(xk1)[2]))
#print("\n")
#print("(Steepest Descent)After stepping:")
#print("---------------:")
#print("xk = " + str(xkFinal2))
#print("numEvals = " + str(iterFinal2))
#print("f(xk) = " + str(fkFinal2))
#print("grad(xk) = " + str(gradkFinal2))
#print("hess(xk) = " + str(hesskFinal2))
#print("\n")
#---------------------------------------------------------------------------------------------------------------



#Steepest Descent Method
#def sDescent(xk, fcn, eps, tol): #eps = eps "epsilon"
#   xklist, fklist = [xk], [fcn(xk)] #list of xk and fk values (fk = fxn(xk))
#   xk0 = xk #saves previous step
#   curIter = 0
#   pk = -fcn(xk)[1]
#   cur = 0
#   stopper = 30 #maximum iterations allowed
#   while cur < stopper :
#      cur = cur + 1
#   #while (abs(pk) > eps).all():
#      #print("pk" + str(abs(pk)))
#      #print("xk" + str(abs(xk)))
#      pk = -fcn(xk)[1]
#      
#      if cur == 15:
#         print(abs(pk))
#      if cur == 29:
#         print(abs(pk))
#      
#      if (abs(pk) <= eps).all():
#         return xk, fcn(xk)[0], fcn(xk)[1], fcn(xk)[2], curIter #found minima
#         break
#      ##Armijo Line Search##
#      gradk = fcn(xk)[1]
#      convVal, iter, alphaConv = Armijo(xk, fcn, pk, gradk, 10**-4, 0.5, 0.7)
#      ######################
#      alpha = alphaConv #saves the found alpha
#      xk1 = xk + np.dot(alpha, pk)
#      xk = xk1
#      curIter = curIter + 1
#   return xk, fcn(xk)[0], fcn(xk)[1], fcn(xk)[2], curIter #found minima


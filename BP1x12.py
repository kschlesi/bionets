# Back Propagation algorithm with one hidden layer
# using conjugate gradient method with Polak-Ribiere update
# with bias node in input layer and hidden layer
# offline updating - update weights after one epoch

# allows for multiple hidden layers, each with the same number of nodes

from __future__ import division
import numpy
import linecache
import math


# activation function: sigmoid
def s(x):
    return 1/(1+numpy.exp(-x))

# derivative of activation function in terms of output y = s(x)
def sp(y):
    return y*(1-y)

class Network:
    # n1  = number input nodes
    # nHn = number of hidden nodes per layer
    # nHl = number of hidden layers
    # n2  = number of output nodes
    def __init__(self,n1,nHl,nHn,n2):		
        self.n1 = n1
        self.nHn = nHn
        self.nHl = nHl
        self.n2 = n2
	self.n  = 2*((n1+1)*nHn  + (nHl-1)*(nHn+1)*nHn + (nHn+1)*n2) 	# total number weights

        # preallocate

        # input I and output O at each layer 
        self.O1  = numpy.array([1.0]*self.n1)
	self.O1E = numpy.array([1.0]*(self.n1+1))

	self.IH  = numpy.zeros((self.nHl,self.nHn))
	self.OH  = numpy.ones((self.nHl,self.nHn))		# matrix whose rows are of output from each hidden layer
	self.OHE = numpy.ones((self.nHl,(self.nHn+1)))

        #self.OH  = numpy.array([1.0]*self.nH)
	#self.OHE = numpy.array([1.0]*(self.nH+1))

        self.I2 = numpy.array([0.0]*self.n2)
        self.O2 = numpy.array([1.0]*self.n2)

	numpy.random.seed(seed=None)


        #############################################
        ###          connection matrices          ###
        #############################################

	#self.C1  = numpy.zeros((self.n1+1,self.nHn))	
        #self.C2  = numpy.zeros((self.nHn+1,self.n2))

        ############# 2 x 6 block ###################
        ### fully connected input ###
	self.C1  = numpy.ones((self.n1+1,self.nHn))

        ### hidden block ###
        self.CH  = numpy.zeros(((self.nHl-1)*(self.nHn+1),self.nHn))

        ## nearest neighbor connections ##

        for k in range(0,self.nHl-1):

            # edges of network #
            #self.CH[k*(self.nHn+1),0:2] = 1
            #self.CH[(k+1)*(self.nHn+1)-2,self.nHn-2:self.nHn] = 1

            # bias layers #
            self.CH[(k+1)*(self.nHn+1)-1,:] = 0

            # intermediate #
            for i in range(1,self.nHn-1):
                self.CH[i+k*(self.nHn+1),i] = 1





        #self.CH[0,0:2] = 1
        #for i in range(1,self.nHn-1):
        #    self.CH[i,i-1:i+2] = 1
        #self.CH[self.nHn-1,self.nHn-2:self.nHn] = 1

        # bias layer: all ones
        #self.CH[self.nHn,:] = 1

        ### fully connected output ###
        self.C2  = numpy.ones((self.nHn+1,self.n2))
        #############################################
        



        # random initial conditions
	self.w1  = numpy.random.uniform(-5,5,(self.C1.shape))	
	self.wH  = numpy.random.uniform(-5,5,(self.CH.shape))	# nHn columns, (nHl-1) blocks of rows, each with (nHn+1) rows
        self.w2  = numpy.random.uniform(-5,5,(self.C2.shape))



        #########################################################
        ### set unconnected weights to zero (for error check) ###
        #########################################################
        for i in range(0,self.C1.shape[0]):
            for j in xrange(0,self.C1.shape[1]):
                if self.C1[i,j]==0:
                    self.w1[i,j]=0

        for i in range(0,self.CH.shape[0]):
            for j in xrange(0,self.CH.shape[1]):
                if self.CH[i,j]==0:
                    self.wH[i,j]=0

        for i in range(0,self.C2.shape[0]):
            for j in xrange(0,self.C2.shape[1]):
                if self.C2[i,j]==0:
                    self.w2[i,j]=0
        #########################################################


        # uniform initial conditions
	#self.w1  = numpy.random.uniform(1.0,1.0,(self.n1+1,self.nH))
        #self.w2  = numpy.random.uniform(1.0,1.0,(self.nH+1,self.n2))
	# perturbed
	#self.w1[self.n1-1,self.nH-1] = numpy.random.uniform(0,1,1)

	self.deltaH = numpy.zeros((self.nHl-1,self.nHn))

	self.w1old = self.w1
	self.wHold = self.wH
	self.w2old = self.w2
 	 	 
  	self.dw1old  = numpy.zeros((self.C1.shape))
	self.dwHold  = numpy.zeros((self.CH.shape)) 
        self.dw2old  = numpy.zeros((self.C2.shape))
 
	self.den1old = 1
	self.denHold = numpy.ones((self.nHl-1))
	self.den2old = 1

	self.update1old = numpy.zeros((self.C1.shape)) 
	self.updateHold = numpy.zeros((self.CH.shape))
	self.update2old = numpy.zeros((self.C2.shape))

        self.updateH = numpy.zeros((self.CH.shape))

	self.E = numpy.array([1.0]*self.n2)

	self.dEdw1 = numpy.zeros((self.C1.shape))
	self.dEdwH = numpy.zeros((self.CH.shape))
	self.dEdw2 = numpy.zeros((self.C2.shape))

	self.dEdw1old = numpy.zeros((self.C1.shape))
	self.dEdwHold = numpy.zeros((self.CH.shape))
	self.dEdw2old = numpy.zeros((self.C2.shape))



####################################################################################################
###                               Forward-Propagate Input                                        ###
####################################################################################################

    def FP(self,x,test,w1,wH,w2):

	if test == True:
	    self.w1 = w1
	    self.wH = wH
	    self.w2 = w2

        ################### output from input layer ######################
       
        self.O1  = numpy.array(x)        	  # output from 1, row length n1
	self.O1E = numpy.append(self.O1,1.0)      # extended output from 1 w/ bias, row length (n1+1)

        ##################################################################


        ############### output from first hidden layer ###################

        self.OH[0]  = s(numpy.dot(self.O1E,self.w1*self.C1)) 		
	self.OHE[0] = numpy.append(self.OH[0],1.0)	     	        

        ##################################################################


        ############ input from kth to (k+1)th hidden layer ##############

	for i in range(1,self.nHl):
	    self.OH[i]  = s( numpy.dot( self.OHE[i-1],self.wH[(i-1)*(self.nHn+1):i*(self.nHn+1)]*self.CH[(i-1)*(self.nHn+1):i*(self.nHn+1)] ) )												
	    self.OHE[i] = numpy.append(self.OH[i],1.0)


        ##################################################################


        ############### final output from output layer ###################

        self.O2 = s(numpy.dot(self.OHE[self.nHl-1],self.w2*self.C2))

        ##################################################################

        return self.O2



####################################################################################################
###                                    Calculate Error                                           ###
####################################################################################################

    def Error(self,y):

        yA = numpy.array(y)
        
        self.E = .5*(yA-self.O2)**2 				# row length n2
        self.Ep = (yA-self.O2)
	#print 'out = ', self.O2

	return self.E




####################################################################################################
###                               Back-Propagate Error                                           ###
####################################################################################################

    def BP(self):


        ########## BP error from output to last hidden layer #############

        delta2 = self.Ep*sp(self.O2)     # row length n2
        self.dEdw2  = numpy.outer(self.OHE[self.nHl-1], delta2)*self.C2 
       
        ##################################################################


        ######## BP error from last to second to last hidden layer #######

	#self.deltaH[self.nHl-2] = numpy.dot(self.w2[:self.nHn],delta2)*sp(self.OH[self.nHl-1])	# row nHn
	#self.dEdwH[((self.nHl-2)*(self.nHn+1)):((self.nHl-1)*(self.nHn+1))] = numpy.outer(self.OHE[self.nHl-2], self.deltaH[self.nHl-2])*self.CH[((self.nHl-2)*(self.nHn+1)):((self.nHl-1)*(self.nHn+1))]

        ##################################################################


        ########## BP error between internal hidden layers ###############

	#for i in range(1,self.nHl-1):
        #    self.deltaH[self.nHl-2-i] = numpy.dot(self.wH[((self.nHl-1-i)*(self.nHn+1)):((self.nHl-i)*(self.nHn+1)-1)],self.deltaH[self.nHl-1-i])*sp(self.OH[self.nHl-1-i])	# row nHn
        #    self.dEdwH[((self.nHl-2-i)*(self.nHn+1)):((self.nHl-1-i)*(self.nHn+1))] = numpy.outer(self.OHE[self.nHl-2-i], self.deltaH[self.nHl-2-i])*self.CH[((self.nHl-2-i)*(self.nHn+1)):((self.nHl-1-i)*(self.nHn+1))]

        ##################################################################



        ######## BP error from first hidden layer to input layer #########

        delta1 = numpy.dot(self.w2[:self.nHn],delta2)*sp(self.OH[0])          	# row length nHn
        self.dEdw1  = numpy.outer(self.O1E,delta1)*self.C1 

        ##################################################################



##################################################################################################################################################
###                                                                                                                                            ###
###                                                     LEARNING ALGORITHM                                                                     ###
###                                                                                                                                            ###
##################################################################################################################################################

    def learn(self,origpts,pseudo,pool,pt,nb,np,w10,wH0,w20,delta=.0001,thresh=.01,tol1=0.001,tol2=.000001,tolB=.0001,kmax=2000,mom=0.5,rate=0.3):
        # original tol1 = 0.000001, delta = 0.000001, thresh = 0.3, tol2=.00000001
        
	
	# pool = number of pseudoitems in pool
	# buff = number of pseudoitems in buffer
	# pseudo is True/False: if True, generate

	if pseudo == True:
	    points  = self.pseudo_buffer(nb,pool,pt,origpts)
	    self.w1 = w10
	    self.wH = wH0
	    self.w2 = w20
        else:
	    points = origpts

	numpy.random.seed(seed=None)
	self.w1old = self.w1
	self.w2old = self.w2
	self.wHold = self.wH
  	self.dw1old  = numpy.zeros((self.n1+1,self.nHn))
	self.dwHold  = numpy.zeros(((self.nHl-1)*(self.nHn+1),self.nHn)) 
        self.dw2old  = numpy.zeros((self.nHn+1,self.n2)) 
	self.den1old = 1
	self.denHold = numpy.ones((self.nHl-1))
	self.den2old = 1
	self.update1old = numpy.zeros((self.n1+1,self.nHn)) 
	self.updateHold = numpy.zeros(((self.nHl-1)*(self.nHn+1),self.nHn))
	self.update2old = numpy.zeros((self.nHn+1,self.n2))
	self.dEdw1 = numpy.zeros((self.n1+1,self.nHn))
	self.dEdwH = numpy.zeros((((self.nHl-1)*(self.nHn+1)),self.nHn))
	self.dEdw2 = numpy.zeros((self.nHn+1,self.n2))
	self.dEdw1old = numpy.zeros((self.n1+1,self.nHn))
	self.dEdwHold = numpy.zeros((((self.nHl-1)*(self.nHn+1)),self.nHn))
	self.dEdw2old = numpy.zeros((self.nHn+1,self.n2))

        q = 0
        for i in range(0,self.C1.shape[0]):
            for j in range(0, self.C1.shape[1]):
                if self.C1[i,j]==1:
                    q = q+1

        for i in range(0,self.CH.shape[0]):
            for j in range(0, self.CH.shape[1]):
                if self.CH[i,j]==1:
                    q = q+1

        for i in range(0,self.C2.shape[0]):
            for j in range(0, self.C2.shape[1]):
                if self.C2[i,j]==1:
                    q = q+1


        print '# = ', q
        print self.w1.shape
        print self.wH.shape
        print self.w2.shape


        print self.C1
        print self.CH
        print self.C2

        print self.w1
        print self.wH
        print self.w2


 

        self.name='1x12';

	w1h  = open('BPdat/w1_'+str(self.name)+'.dat','a')
        w2h  = open('BPdat/w2_'+str(self.name)+'.dat','a')
	wHh  = open('BPdat/wH_'+str(self.name)+'.dat','a')
        jh   = open('BPdat/iter_'+str(self.name)+'.dat','a')
	Eh   = open('BPdat/error_'+str(self.name)+'.dat','a')
	Eopph = open('BPdat/Eopp_'+str(self.name)+'.dat','a')
        Enpph = open('BPdat/Enpp_'+str(self.name)+'.dat','a')
        Ebpph = open('BPdat/Ebpp_'+str(self.name)+'.dat','a')
	Enh  = open('BPdat/En_'+str(self.name)+'.dat','a')
	Ebh  = open('BPdat/Eb_'+str(self.name)+'.dat','a')
	Eoh  = open('BPdat/Eo_'+str(self.name)+'.dat','a')
	x0h  = open('BPdat/x0.dat','a')
	y0h  = open('BPdat/y0.dat','a')
	xph  = open('BPdat/x_'+str(self.name)+'.dat','a')
	yph  = open('BPdat/y_'+str(self.name)+'.dat','a')
	newh = open('BPdat/newpt_'+str(self.name)+'.dat','a')
        resbh = open('BPdat/resb_'+str(self.name)+'.dat','a')
        resnh = open('BPdat/resn_'+str(self.name)+'.dat','a')
        resoh = open('BPdat/reso_'+str(self.name)+'.dat','a')
        res0h = open('BPdat/res0_'+str(self.name)+'.dat','a')

	if pseudo == True:
	    for i in range(0,points.shape[0]):
	    	xph.write("%s " % points[i,0])
		yph.write("%s " % points[i,1])
	    
	    newh.write("%s " % points[nb,0])
	    newh.write("%s " % points[nb,1])
	    newh.write("\n")
	    xph.write("\n")
	    yph.write("\n")

	else:
	    for i in range(0,points.shape[0]):
	    	x0h.write("%s " % points[i,0])
		y0h.write("%s " % points[i,1])
	    x0h.write("\n")
	    y0h.write("\n")
	

	ET    = 1
	Ept   = 1
	ETold = 0
	j = 0		# iteration index
	k = 0
        a = 0


	if pseudo == False:
	    iter_max = 100000
            # orig: iter_max = 1000000
            
	else:
	    iter_max = 500
	    # orig:  iter_max = 500


	#while True:
        mm = 2
	for j in xrange(iter_max):
            if mm==1:
                print 'what??'
	    else:
		Enew = 0
		Eold = 0

                ####################################################
                ##                 FINAL OUTPUT                   ##
                ####################################################

		#if Ept < tol1:
                #if ET < tol1:

                ############## stop conditions #####################
		if ET < tol1 or k > 1000 or j == iter_max-1 or numpy.isnan(ET):


                    Eb = 0
                    En = 0
                    Eo = 0

                    if numpy.isnan(ET):
                        ET = ETold
                        for i in range(0,self.w1.shape[0]):
                            for j in range(0,self.w1.shape[1]):
                                self.w1[i,j] = self.w1old[i,j]

                        for i in range(0,self.wH.shape[0]):
                            for j in range(0,self.wH.shape[1]):
                                self.wH[i,j] = self.wHold[i,j]

                        for i in range(0,self.w2.shape[0]):
                            for j in range(0,self.w2.shape[1]):
                                self.w2[i,j] = self.w2old[i,j]

                    else:
                        ETold = ET


                    if pseudo == True:

                        ##################################
                        ##        BUFFERED ERROR        ##
                        ##################################

                        for i in range(0,nb):
                            xb = points[i,0]
                            yb = points[i,1]

                            out_b = self.FP(xb,False,0,0,0)
                            Eb = Enew + self.Error(yb)   ## error in buffered points only
                            resb = self.Ep
                            resbh.write("%-14f " % resb)
                        resbh.write("\n")

                        ##################################
                        ##          NEW ERROR           ##
                        ##################################
		    
                        for i in range(nb,points.shape[0]):
                            xn = points[i,0]
                            yn = points[i,1]

                            out_n = self.FP(xb,False,0,0,0)
                            En = En + self.Error(yn)     ## error in new points only
                            resn = self.Ep
                            resnh.write("%-14f " % resn)
                        resnh.write("\n")
                        
                    ##################################
                    ##         ORIGINAL ERROR       ##
                    ##################################

		    for i in range(0,origpts.shape[0]):
		    	xo = origpts[i,0]
                     	yo = origpts[i,1]

			out_o = self.FP(xo,False,0,0,0)
			Eo = Eo + self.Error(yo)      ## error in original points only

                        reso = self.Ep

                        if pseudo == True:
                            resoh.write("%-14f " % reso)
                        else:
                            res0h.write("%-14f " % reso)
                    resoh.write("\n")
                    res0h.write("\n")


                    ##################################
                    ##          TOTAL ERROR         ##
                    ##################################

                    if pseudo == True:
                        ## per point errors
                        Ebpp  = Eb/(nb)                
                        Enpp  = En/(points.shape[0]-nb)     
                        Eopp  = Eo/(origpts.shape[0])

                    ## total errors
                    Etb = Eb + En
                    Eto = Eo + En

		    if pseudo == True:
			Ebpph.write("%-14f " % Ebpp)
			Enpph.write("%-14f " % Enpp)
			Eopph.write("%-14f " % Eopp)
			Eoh.write("%-14f " % Eo)
			Ebh.write("%-14f " % Eb)
			Enh.write("%-14f " % En)
                        Eh.write("%-14f\n " % Etb)
                    else:
                        Eh.write("%-14f\n " % Eo)
			Eoh.write("%-14f " % Eo)

		    w1h.write("%s\n" % self.w1)
		    w2h.write("%s\n" % self.w2)
                    wHh.write("%s\n" % self.wH)
		    jh.write("%s\n" % j)


		    w1h.close()
		    w2h.close()
                    wHh.close()
		    jh.close()
		    Eh.close()
		    Eopph.close()
		    Enpph.close()
                    Ebpph.close()
		    Enh.close()
		    Eoh.close()
		    Ebh.close()
		    x0h.close()
		    y0h.close()
		    xph.close()
		    yph.close()
		    newh.close()
                    resbh.close()
                    resnh.close()
                    resoh.close()
                    res0h.close()



		    print 'final error = %-14f' % Et
		    break



                #################################################################
                ##                CONJUGATE GRADIENT DESCENT                   ##
                #################################################################

	        else:
	
	    	    #ETold = ET
		    dw1  = 0
		    dw2  = 0
                    dwH  = 0
		    for i in range(0,points.shape[0]):
		    	x = points[i,0]
                     	y = points[i,1]
                     	self.FP(x,False,0,0,0)
			Enew = Enew + self.Error(y)
                     	self.BP()

			dw2 = dw2+self.dEdw2
			dw1 = dw1+self.dEdw1
  
                    


		    if pseudo == False:
			Ept = Enew
		    else:
			ypt = self.FP(points[0,0],False,0,0,0)
			Ept = .5*(points[0,1]-ypt)**2	 	

		    k = k+1
		    if abs(Enew-ET) > tol2:
		    	k = 0

		    ################## determine Polak Ribiere CG direction ####################	

		    a2 = 0
		    p2 = dw2*(dw2-self.dw2old)

                    ## sum over all elements in p2 ##
		    for m in p2:
			for n in range(0,self.C2.shape[1]):
			    a2 = a2+m[n]


		    den2 = 0
		    for m in dw2:
		     	for n in range(0,self.n2):
			    den2 = den2 + m[n]*m[n]

		    beta2 = a2/self.den2old
		    self.den2old = den2

		    if beta2<0:
			beta2 = 0 



		    aH = numpy.zeros((self.nHl-1))
		    pH = dwH*(dwH-self.dwHold)

		    for m in range(0,self.nHl-1):
                        mi = m*(self.nHn+1)
                        mf = (m+1)*(self.nHn+1)
			for n in range(mi,mf):
                            for p in range(0,self.nHn):
                                aH[m] = aH[m]+pH[n,p]


                    denH = numpy.zeros((self.nHl-1))
		    for m in range(0,self.nHl-1):
                        mi = m*(self.nHn+1)
                        mf = (m+1)*(self.nHn+1)
			for n in range(mi,mf):
                            for p in range(0,self.nHn):
                                denH[m] = denH[m]+dwH[n,p]*dwH[n,p]


		    betaH = aH/self.denHold
		    self.denHold = denH

                    for m in range(0,self.nHl-1):
                        if betaH[m]<0:
                            betaH[m] = 0 

		    a1 = 0
		    #p1 = self.dEdw1*(self.dEdw1-self.dEdw1old)
		    p1 = dw1*(dw1-self.dw1old)
		    for m in p1:
			for n in range(0,self.nHn):
			    a1 = a1+m[n]

		    den1 = 0
		    #for m in self.dEdw1:
		    for m in dw1:
		     	for n in range(0,self.nHn):
			    den1 = den1 + m[n]*m[n]
			    #print 'm[n] = ',m[n]


		    beta1 = a1/self.den1old
		    self.den1old = den1


		
		    if beta1 < 0:
			beta1 = 0

		    # restart every n = (n1+1)*nH+(nH+1)*n2 iterations: move in gradient direction
	            #if j % self.n == 0:
		    #	beta1 = 0
		    #	beta2 = 0
                    #    betaH = numpy.zeros(self.nHl-1)

	            if k % 50 == 0:
		    	beta1 = 0
		    	beta2 = 0
                        betaH = numpy.zeros(self.nHl-1)


		    # restart if beta is below threshold
		    #if beta1 < thresh:	
			#beta1 = 0
		    #if beta2 < thresh:
			#beta2 = 0
                    #for m in range(0,self.nHl-1):
                        #if betaH[m] < thresh:
                            #betaH[m] = 0


		    self.update2 = dw2+beta2*self.update2old
		    self.update1 = dw1+beta1*self.update1old

                    #print 'update1 = ', self.update1, '\n'
                    #print 'update2 = ', self.update2, '\n'

                    for m in range(0,self.nHl-1):
                        self.updateH[m*(self.nHn+1):(m+1)*(self.nHn+1)] = dwH[m*(self.nHn+1):(m+1)*(self.nHn+1)]+betaH[m]*self.updateHold[m*(self.nHn+1):(m+1)*(self.nHn+1)]

                    #print 'updateH = ', self.updateH, '\n'

		    self.update2old = self.update2
		    self.update1old = self.update1
                    self.updateHold = self.updateH

		    ############### line search in CG direction #######################

	   	    lambda_a = 0 
		    lambda_b = 0
		    lambda_c = 0
		    z = 0
		    Eb = Enew	
		    Ec = Enew
		    Ec = Enew

		    update1_test = self.update1
		    update2_test = self.update2			
		    updateH_test = self.updateH

		    while True:

			Et = 0

			if z > 50:
			    bracket = False
			    break

			#if numpy.isnan(En):
			#    print 'trial error NaN'
			#    break


  			lambda_c = lambda_b+(2**z)*delta
		    	self.w1 = self.w1old+lambda_c*update1_test
		    	self.w2 = self.w2old+lambda_c*update2_test
			self.wH = self.wHold+lambda_c*updateH_test
		    	for i in range(0,points.shape[0]):
			    xt = points[i,0]
			    yt = points[i,1]
			    self.FP(xt,False,0,0,0)

		    	    Et = Et+self.Error(yt)
			Ec = Et

			if Ec > Eb:
			    if z > 0:	##### bracket #####
				#lambda_update = lambda_b
				bracket = True
			    	break
			    #elif z==1:	##### reset cg driection #####
				#lambda_update = lambda_b
				#print lambda_a, lambda_b, lambda_c, Ec-Eb
				#bracket = False
				#break

			    else:
				#lambda_update = 1
				bracket = False
				break

		    	    #	update2_test = dw2+beta2*update2_test
		    	    #	update1_test = dw1+beta1*update1_test

                    	    #	for m in range(0,self.nHl-1):
                            #	    updateH_test[m*(self.nHn+1):(m+1)*(self.nHn+1)] = dwH[m*(self.nHn+1):(m+1)*(self.nHn+1)]+betaH[m]*updateH_test[m*(self.nHn+1):(m+1)*(self.nHn+1)]

				#self.update1old = self.update1
				#self.update2old = self.update2
				#self.updateHold = self.updateH
			else:
	  		    Ea = Eb
			    Eb = Ec
			    lambda_a = lambda_b
			    lambda_b = lambda_c
			    z = z+1


		    ############# begin bracketing ####################

		    b = 0
		    while True:

			if bracket == False:
			    break

			gold = (3-numpy.sqrt(5))/2.

			if numpy.abs(lambda_c-lambda_a) < tolB:
			    break

			if b > 100:
			    break
	
			

			else:
			    ################ first bracket ################################
			    if numpy.abs(lambda_b-lambda_a) < numpy.abs(lambda_c-lambda_b):
			    	lambda_test = lambda_b+gold*numpy.abs(lambda_c-lambda_b)
			    	self.w1 = self.w1old+lambda_test*self.update1
			    	self.w2 = self.w2old+lambda_test*self.update2
			    	self.wH = self.wHold+lambda_test*self.updateH
				b = b+1

			    	Etest = 0
			    	for i in range(0,points.shape[0]):
				    xt = points[i,0]
				    yt = points[i,1]
				    self.FP(xt,False,0,0,0)
		
	   	    	    	    Etest = Etest+self.Error(yt)

			    	if Etest < Eb:
				    lambda_a = lambda_b
				    lambda_b = lambda_test 

			    	else:
				    lambda_c = lambda_test

			    ################ second bracket ################################
			    else:
				lambda_test = lambda_b-gold*numpy.abs(lambda_b-lambda_a)
			    	self.w1 = self.w1old+lambda_test*self.update1
			    	self.w2 = self.w2old+lambda_test*self.update2
			    	self.wH = self.wHold+lambda_test*self.updateH
				b = b+1

			    	Etest = 0
			    	for i in range(0,points.shape[0]):
				    xt = points[i,0]
				    yt = points[i,1]
				    self.FP(xt,False,0,0,0)
	
	   	    	    	    Etest = Etest+self.Error(yt)

			        if Etest < Eb:
				    lambda_c = lambda_b
				    lambda_b = lambda_test 

			        else:
				    lambda_a = lambda_test
			
		    ############# update with value from center of bracket ####################	 

		    if z == 0 or z > 50:
			lambda_update = delta/4.
                        for i in range(0,self.update1.shape[0]):
                            for l in range(0,self.update1.shape[1]):
                                self.update1[i,l] = dw1[i,l]
                        for i in range(0,self.updateH.shape[0]):
                            for l in range(0,self.updateH.shape[1]):
                                self.updateH[i,l] = dwH[i,l]
                        for i in range(0,self.update2.shape[0]):
                            for l in range(0,self.update2.shape[1]):
                                self.update2[i,l] = dw2[i,l]
		    else:
		    	lambda_update = lambda_b
		
		    self.w1 = self.w1old+lambda_update*self.update1
		    self.w2 = self.w2old+lambda_update*self.update2
		    self.wH = self.wHold+lambda_update*self.updateH

		    self.dEdw2old = self.dEdw2
		    self.dEdw1old = self.dEdw1
                    self.dEdwHold = self.dEdwH

		    self.w1old = self.w1
		    self.w2old = self.w2
		    self.wHold = self.wH

        	    self.dw1old = dw1
        	    self.dw2old = dw2
		    self.dwHold = dwH



	    	    ET = Enew
		    #if pseudo == True:
		    #	print 'Efinal = ', ET


	    if j % 10 == 0:
		print 'j = ', j, '  error = %-14f' % ET	, 'k = ', k, 'z = ', z, 'b = ', b
                if pseudo == True:
                    Eh.write("%-14f " % ET)

            if j % 100 == 0:
                if pseudo == False:
                    Eh.write("%-14f " % ET)

	    j = j+1
			
################################################################################################


##########################################################
    def w1(self):
	return self.w1

    def wH(self):
	return self.wH

    def w2(self):
	return self.w2
##########################################################


##########################################################
    def test(self,bin,pseudo,new_W,w1,wH,w2):
	self.xtest = numpy.zeros(bin+1)
	self.ytest = numpy.zeros(bin+1)

	xh = open('BPdat/in.dat','w')
	y0h = open('BPdat/out0_'+str(self.name)+'.dat','a')
	yh = open('BPdat/out_'+str(self.name)+'.dat','a')
        varh = open('BPdat/var_'+str(self.name)+'.dat','a')
        avgh = open('BPdat/avg_'+str(self.name)+'.dat','a')

	for i in range(0,bin+1):
	    self.xtest[i] = self.xtest[i] + (i)*(1./bin)
	    xh.write("%s " % self.xtest[i])
	xh.write("\n")
	    
        var = 0
        avgy = 0
	if pseudo == True:
	    for j in range(0,bin+1):
	    	self.ytest[j]=self.FP(self.xtest[j],new_W,w1,wH,w2)
	    	avgy = avgy + self.ytest[j]/bin
                yh.write("%s " % self.ytest[j])

            for j in range(0,bin+1):
                var = var + (self.ytest[j]-avgy)**2/bin
            

            avgh.write("%s " % avgy)
            varh.write("%s " % var)
	    yh.write("\n")
	else:
	    for j in range(0,bin+1):
	    	self.ytest[j]=self.FP(self.xtest[j],new_W,w1,wH,w2)
	    	y0h.write("%s " % self.ytest[j])
	    y0h.write("\n")
	
	xh.close()
	y0h.close()
	yh.close()
        varh.close()
        avgh.close()

	return self.ytest
##########################################################


##########################################################
    def fnc(self,x):
	#return 6*(0.8*x**7 - 2.5*x**6 + 2.5*x**5 - 3.8*x**4 + 5.8*x**3 - 3.5*x**2 + 0.8*x)
	#return 1.5*x**3 - 2*x**2 + x
	#return 4*x**3 - 6*x**2 + 2*x + 0.5

	#return 5*x**3 - 7*x**2 + 2*x + 0.5

	#return 4*(x-x**2)
	#return -2*x**2 + 2*x + 0.3
	#return x
	
        return .1*numpy.sin(10*x)**2+.4*numpy.sin(30*x)**2+0.5*numpy.sin(18*x)**2
##########################################################


##########################################################
    def pseudo_pool(self,bin,new_W,w1,wH,w2):
	#xpt = pt[0,0]
	#self.input  = numpy.random.uniform(0,1,bin)

        l_offset = 0.1
        r_offset = 0.1
        #if xpt < l_offset:
        #    l_offset = 0 
        #elif xpt > 1.0-r_offset:
        #    r_offset = 0

	self.input  = numpy.random.uniform(0,1,bin)
	#self.input  = numpy.random.uniform(0,xpt-l_offset,int(xpt*bin))
	#self.input  = numpy.append(self.input,numpy.random.uniform(xpt+r_offset,1,int((1-xpt)*bin)))

	self.output = numpy.zeros(bin)
	#self.output = numpy.zeros(int(xpt*bin)+int((1-xpt)*bin))

	self.pool = numpy.zeros((bin,2))
	#self.pool = numpy.zeros((int(xpt*bin)+int((1-xpt)*bin),2))

	for i in range(0,bin):
	#for i in range(0,int(xpt*bin)+int((1-xpt)*bin)):
	    self.output[i] = self.FP(self.input[i],new_W,w1,wH,w2)
	    self.pool[i,0] = self.input[i]
	    self.pool[i,1] = self.output[i]

	return self.pool
##########################################################


##########################################################
    def new_pt(self,nnew):
        numpy.random.seed(seed=None)
	self.new = numpy.zeros((nnew,2))
        for i in range(0,nnew):
            self.new[i,0] = numpy.random.uniform(0,1)
            self.new[i,1] = numpy.random.uniform(0,1)
            # original
            #self.new[i,1] = numpy.random.uniform(0,1)
	#self.new[0,1] = self.fnc(self.new[0,0])
	#self.new[0,1] = self.FP(self.new[0,0])

	return self.new
##########################################################


##########################################################
    def pseudo_buffer(self,bin,pool,newpt,origpts):
	numpy.random.seed(seed=None)
	b = numpy.zeros(bin)
	#self.buff = numpy.zeros((bin+3,2))
        nnew = newpt.shape[0]
        self.buff = numpy.zeros((bin+nnew,2))
	for i in range(0,bin):
	    b[i] = numpy.random.randint(0,pool.shape[0])
	    #for j in range(0,i):
		#if b[i] == b[j]:
		    #b[i] = numpy.random.randint(-1,pool.shape[0])

	####  fixed endpoints ######
	#self.buff[0,0]     = origpts[0,0]
	#self.buff[0,1]     = origpts[0,1]
	#self.buff[bin+1,0] = origpts[origpts.shape[0]-1,0]
	#self.buff[bin+1,1] = origpts[origpts.shape[0]-1,1]

	#for i in range(0,bin):
	#    self.buff[i+1,0]  = pool[b[i],0]
	#    self.buff[i+1,1]  = pool[b[i],1]

        #for i in range(0,nnew):
        #    self.buff[bin+2+i,0] = newpt[i,0]
        #    self.buff[bin+2+i,1] = newpt[i,1]


	####  free endpoints ######
	for i in range(0,bin):
	    self.buff[i,0]  = pool[b[i],0]
	    self.buff[i,1]  = pool[b[i],1]

        for i in range(0,nnew):
            self.buff[bin+i,0] = newpt[i,0]
            self.buff[bin+i,1] = newpt[i,1]

	#self.buff[bin,0] = self.new_pt[0,0]
	#self.buff[bin,1] = self.new_pt[0,1]

	return self.buff
##########################################################	
	

##########################################################
    def err(self,points,new_W,w1,wH,w2):
	ef = open('BPdat/errorf_'+str(self.name)+'.dat','a')
	final_err = 0
	for i in range(0,points.shape[0]):
	    x = points[i,0]
	    y = points[i,1]
	    self.FP(x,new_W,w1,wH,w2)
	    final_err = final_err + self.Error(y)

	ef.write("%s " % final_err)
	ef.close()
	return final_err
##########################################################

def run(iter):

        net = Network(1,1,12,1)

        points = numpy.array([#[0.,net.fnc(0.)],
			      [.1,net.fnc(.1)],
			      [.26,net.fnc(.26)],
			      [.42,net.fnc(.42)],
			      [.58,net.fnc(.58)],
			      [.74,net.fnc(.74)],
			      [.9,net.fnc(.9)],
				])


	
        pool = 0
        pt = 0
	nb = 6     # number in buffer (excluding endpoints)
	np = 1000  # number in pool
        nnew = 6  # number of new points
   	

	net.learn(points,False,pool,pt,nb,np,0,0,0)
	w10 = net.w1
	wH0 = net.wH
	w20 = net.w2

	net.test(100,False,False,0,0,0)
        pool = net.pseudo_pool(np,False,0,0,0)

	for i in range(1,iter+1):

	    pt = net.new_pt(nnew)

	    net.learn(points,True,pool,pt,nb,np,w10,wH0,w20)
	    print 'iteration = ', i
	    net.test(100,True,False,0,0,0)
	    net.err(points,False,0,0,0)




if __name__ == '__main__':
    run(1000)


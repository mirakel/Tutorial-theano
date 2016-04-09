import theano
import theano.tensor as T
import random
import numpy

x = T.vector()
#declaracion de variables compartidas
w = theano.shared(numpy.array([1.,1.])) #valor inicil 1.0, 1.0
b = theano.shared(0)

z = T.dot(w,x) + b
y = 1 / (1 + T.exp(-z))

neuron = theano.function(    #esta funcion puede acceder a la variables compartidas
			inputs = [x],
			outputs = y)  

#se puede acceder y cambiar los valores de las variables compartidas con get_value() y set_value()

print w.get_value()

w.set_value([-1,1])

for i in range(100):
	x = [random.random(), random.random()]
	print x, neuron(x)
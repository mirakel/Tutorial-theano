import theano
import theano.tensor as T
import numpy
import random

x = T.vector()
w = theano.shared(numpy.array([-1.,1.]))
b = theano.shared(0.)

z = T.dot(w,x) + b
y = 1 / (1 + T.exp(-z))

neuron = theano.function(
			inputs = [x],
			outputs = y)

y_hat = T.scalar()  #referencia de variable salida
cost = T.sum((y - y_hat) ** 2)  #funcion costo

dw, db = T.grad(cost,[w,b])  #gradiente con respecto a w y b

gradient = theano.function(   #Funcion para calcular gradientes
			inputs = [x,y_hat],
			outputs = [dw,db])

x = [1, -1]
y_hat = 1

for i in range(100):
	print neuron(x)
	dw, db = gradient(x, y_hat) 
	#Actualizando los pesos y el bias
	w.set_value(w.get_value() - 0.1 * dw)
	b.set_value(b.get_value() - 0.1 * db)
	print w.get_value(), b.get_value()
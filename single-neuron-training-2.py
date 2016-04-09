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
			updates = [(w, w - 0.1 * dw),(b, b - 0.1 * db)])  #(variable compartida, expresion), reemplaza el valor de la variable compartida por la expresion

x = [1, -1]
y_hat = 1

for i in range(100):
	print neuron(x)
	gradient(x, y_hat) 
	print w.get_value(), b.get_value()
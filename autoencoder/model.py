from neuralpy.models import Sequential
from neuralpy.layers.linear import Dense
from neuralpy.layers.activation_functions import ReLU, Sigmoid
from neuralpy.loss_functions import MSELoss
from neuralpy.optimizer import Adam
model = Sequential()

# Encoder
model.add(Dense(n_inputs=(28*28), n_nodes=128))
model.add(ReLU())
model.add(Dense(n_inputs=128, n_nodes=64))
model.add(ReLU())
model.add(Dense(n_inputs=64, n_nodes=32))
model.add(ReLU())
model.add(Dense(n_inputs=32, n_nodes=16))

# Decoder

model.add(Dense(n_inputs=16, n_nodes=32))
model.add(ReLU())
model.add(Dense(n_inputs=32, n_nodes=64))
model.add(ReLU())
model.add(Dense(n_inputs=64, n_nodes=128))
model.add(ReLU())
model.add(Dense(n_inputs=128, n_nodes=(28*28)))
model.add(Sigmoid())

print(model)

model.build()
model.compile(optimizer=Adam(), loss_function=MSELoss())

print(model.summary())

from absl import logging
import tensorflow as tf
import keras


# keras mette a disposizioni 3 api per poter sviluppare reti, uno di questi e' il subclassing
# la classe base che si va a derivare e' keras.layers.Layer
#  class Layer(BackendLayer, Operation, KerasSaveable):
#  __init__ -> stato creato con self.add_weight, che ti fa aggiungere dei parametri associato ad un nome
#              super init prender activity_regularizer, trainable, dtype, autocast, name, tra cui
#              Trainable -> allenabile, da calcolare gradiente, dtype -> O tipo dato o keras.DTypePolicy
# una volta inizializzato, il layer  ha name, dtype, trainable_weights (list), non_trainable_weights,
#   la lcro concatenazione weights: [trainable_weights..., non_trainable_weights...]
#   trainable -> se i trainable_weights vanno allenati o meno
#   input_spec -> condizioni per validare l'input alla funzione call
# build -> funzione chiamata prima della prima call con una data input size, al fine di costruire il
#          grafo di computazione. Mentre init inizializza i layers che non dipendono dalla size dell'input, build
#          inizializza i layers che dipendono dalla size dell'input
# call -> chiamata in __call__. dopo build, oltre all'input, prende keyword arguments opzionali tra cui
#  1 - training (bool) - usare training mode (gradient) o inference mode
#  2 - mask (tensore di bool) - maschera di timestamp dell'input (per layers RNN)
# get_config -> dict contenente la configurazione da usare per inizializzare il layer (la passi a __init__). se
#               la configurazione differisce dai parametri passati a __init__, allora override from_config(self)
class Linear(keras.layers.Layer):
    def __init__(self, *, units: int = 32, input_dim: int = 32) -> None:
        # activity_regularizer = None, trainable = True, name = None (usera il self.__class__.__name__)
        # dtype = None (usera' keras.backend.floatx() == float32)
        super().__init__()

        # aggiunge questa in trainable_weights
        self.w = self.add_weight(
            shape=(input_dim, units),  # shape del tensore di parametri, default (), cioe' scalar
            initializer="random_normal",  # init casuale. default "global_uniform" (floats) o "zeros" (int, bool, ...)
            trainable=True,  # valore default
            autocast=False,  # dice se devi fare il cast del dtype nella call (default True)
            # regularizer - se applicare weight decay all'ottimizzazione del parametro
            # constraint - oggetto da chiamare dopo ogni update, o stringa di un constraint built-in, (def. None)
            name="weights",  # per debugging
            # aggregation - 'mean' (def.), 'sum', 'only_first_replica' - passato a __init__ di backend.Variable
            #               specifica come un variabile distributita in piu dispositivi deve essere aggregata
        )
        self.b = self.add_weight(
            shape=(units,), initializer="zeros", trainable=True, autocast=False, name="bias"
        )
        logging.info(f"the trainable layers are {''.join([str(weight.shape) for weight in self.trainable_weights ])}")

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return keras.ops.matmul(inputs, self.w)


# esempio di una rete senza trainable parameters che accumula tensore di somme ad ogni chiamata di call
class ComputeSum(keras.layers.Layer):
    def __init__(self, *, input_dim: int = 32) -> None:
        super().__init__()
        self.total = self.add_weight(
            shape=(input_dim,), initializer="zeros", trainable=False, autocast=False, name="total"
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.total.assign_add(keras.ops.sum(inputs, axis=0))
        return self.total


# puoi comporre piu layers dentro layers, e l'outer layer tracka in automatico tutti i parametri all'interno dei
# layer figli
class MLPBlock(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(units=32, input_dim=32)
        self.linear2 = Linear(units=32, input_dim=32)
        self.linear3 = Linear(units=1, input_dim=32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.linear1(inputs)
        x = keras.activations.relu(x)
        x = self.linear2(x)
        x = keras.activations.relu(x)
        return self.linear3(x)  # ritorno logits, serve sigmoide per probabilita'


# i moduli fondamentali sono keras.ops, keras.activations, keras.random, keras.layers, che sono backend-agnostic


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
        # la funzione add_loss puo' essere chiamata dall'outer layer per iniettare nel grafo di computazione
        # la loss sull'output, e nei singoli layers costituenti nel caso si voglia applicare della regolarizzazione
        # essendo nella classe che implementa un singolo layer, qui facciamo la L2 Regularization
        # quando si fa backprop, il contenuto add_loss verra' aggiunto all'upstream loss provenienti dagli altri rami
        # del grafo di computazione
        output = keras.ops.matmul(inputs, self.w)
        gamma = 1e-2
        self.add_loss(keras.ops.sum(keras.ops.square(self.w)) * gamma)
        return output


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
        logits = self.linear3(x)  # ritorno logits, serve sigmoide(binaria),softmax(naria) per probabilita'

        # cross entropy loss
        # self.add_loss(keras.ops.mean(keras.losses.categorical_crossentropy(
        #    y_true=<nel ciclo di training, hai i labels a disposizione>,
        #    y_pred=keras.ops.nn.softmax(logits),
        # )))

        return logits


# i moduli fondamentali sono keras.ops, keras.activations, keras.random, keras.layers, che sono backend-agnostic
class MLPTrainer:
    def __init__(self, *, lr: float = 1e-3, batch_size: int = 32, epochs: int = 3):
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self, model: keras.layers.Layer, x_train: tf.Tensor, y_train: tf.Tensor, x_val: tf.Tensor, y_val: tf.Tensor) -> None:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        for epoch in self.epochs:
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                # chiamare modello in gradienttape fa si che i gradienti siano automaticamente calcolati
                # tenendo in consideraizone solo i trainable_weights con le loro add_losses
                with tf.GradientTape() as tape:
                    logits = model(x_batch_train, training=True)
                    loss_value = self.loss_fn(y_batch_train, logits)

                # preleva i gradienti per un sottoinsieme dei trainable weights dal tape (in questo caso ho preso tutto)
                grads = tape.gradient(loss_value, model.trainable_weights)

                # applicare il gradiente secondo la logica specificata dal optimizer
                self.optimizer.apply(grads, model.trainable_weights)

                # logga ogni 100 steps
                if step % 100 == 0:
                    logging.info(f"")
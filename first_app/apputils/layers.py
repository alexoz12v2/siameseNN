from absl import logging
import tensorflow as tf
import time
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
            shape=(
                input_dim,
                units,
            ),  # shape del tensore di parametri, default (), cioe' scalar
            initializer="random_normal",  # init casuale. default "global_uniform" (floats) o "zeros" (int, bool, ...)
            trainable=True,  # valore default
            # regularizer - se applicare weight decay all'ottimizzazione del parametro
            # constraint - oggetto da chiamare dopo ogni update, o stringa di un constraint built-in, (def. None)
            name="weights",  # per debugging
            # aggregation - 'mean' (def.), 'sum', 'only_first_replica' - passato a __init__ di backend.Variable
            #               specifica come un variabile distributita in piu dispositivi deve essere aggregata
        )
        self.b = self.add_weight(
            shape=(units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )
        logging.info(
            "The trainable layers are %s",
            "".join(str(weight.shape) for weight in self.trainable_weights),
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        # la funzione add_loss puo' essere chiamata dall'outer layer per iniettare nel grafo di computazione
        # la loss sull'output, e nei singoli layers costituenti nel caso si voglia applicare della regolarizzazione
        # essendo nella classe che implementa un singolo layer, qui facciamo la L2 Regularization
        # quando si fa backprop, il contenuto add_loss verra' aggiunto all'upstream loss provenienti dagli altri rami
        # del grafo di computazione
        output = tf.math.add(tf.linalg.matmul(inputs, self.w), self.b)
        gamma = 1e-2
        self.add_loss(tf.math.reduce_sum(tf.math.square(self.w)) * gamma)
        self.add_loss(tf.math.reduce_sum(tf.math.square(self.b)) * gamma)
        return output


# esempio di una rete senza trainable parameters che accumula tensore di somme ad ogni chiamata di call
class ComputeSum(keras.layers.Layer):
    def __init__(self, *, input_dim: int = 32) -> None:
        super().__init__()
        self.total = self.add_weight(
            shape=(input_dim,),
            initializer="zeros",
            trainable=False,
            autocast=False,
            name="total",
        )

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        self.total.assign_add(tf.math.reduce_sum(inputs, axis=0))
        return self.total


# puoi comporre piu layers dentro layers, e l'outer layer tracka in automatico tutti i parametri all'interno dei
# layer figli
class MLPBlock(keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = Linear(units=32, input_dim=32)
        self.linear2 = Linear(units=32, input_dim=32)
        self.linear3 = Linear(units=10, input_dim=32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        x = self.linear1(inputs)
        x = keras.activations.relu(x)
        x = self.linear2(x)
        x = keras.activations.relu(x)
        logits = self.linear3(
            x
        )  # ritorno logits, serve sigmoide(binaria),softmax(naria) per probabilita'

        return logits


# i moduli fondamentali sono keras.ops, keras.activations, keras.random, keras.layers, che sono backend-agnostic
class MLPTrainer:
    def __init__(self, *, lr: float = 1e-2, batch_size: int = 32, epochs: int = 3):
        self.optimizer: keras.optimizers.Optimizer = keras.optimizers.Adam(
            learning_rate=lr
        )
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.batch_size = batch_size
        self.epochs = epochs
        self.train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

    # Monitoraggio processo di training https://keras.io/guides/writing_a_custom_training_loop_in_tensorflow/#a-first-endtoend-example
    # 1. chiama metric.update_state() dopo ogni batch
    # 2. chiama metric.result()       quando mostri il valore corrente della metrica
    # 3. chiama metric.reset_state()  quando devi resettare la metrica (alla fine di una epoch)
    def train(
        self,
        model: keras.layers.Layer,
        x_train: tf.Tensor,
        y_train: tf.Tensor,
        x_val: tf.Tensor,
        y_val: tf.Tensor,
    ) -> None:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(self.batch_size)

        for epoch in range(self.epochs):
            logging.info("-" * 40)
            logging.info("Start of epoch %d", epoch)
            logging.info("-" * 40)
            start_time = time.time()
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                loss_value = self._train_step(model, x_batch_train, y_batch_train)
                # chiamare modello in gradienttape fa si che i gradienti siano automaticamente calcolati
                # tenendo in consideraizone solo i trainable_weights con le loro add_losses
                # with tf.GradientTape() as tape:
                #    logits = model(x_batch_train, training=True)
                #    loss_value = self.loss_fn(y_batch_train, logits)

                ## preleva i gradienti per un sottoinsieme dei trainable weights dal tape (in questo caso ho preso tutto)
                # grads = tape.gradient(loss_value, model.trainable_weights)

                ## applicare il gradiente secondo la logica specificata dal optimizer
                # self.optimizer.apply(grads, model.trainable_weights)

                ## aggiorna metrica
                # self.train_acc_metric.update_state(y_batch_train, logits)

                # logga ogni 100 steps
                if step % 100 == 0:
                    logging.info(
                        "Training Loss (for 1 batch) at step %d: %.4f",
                        step,
                        float(loss_value),
                    )
                    logging.info(
                        "Seen so far: %d samples", (step + 1) * self.batch_size
                    )

            # mostra la accuracy per la epoca corrente
            train_acc = self.train_acc_metric.result()
            logging.info("Training acc over epoch: %.4f", float(train_acc))

            # resetta la metrica prima della prossima epoca
            self.train_acc_metric.reset_state()

            # validation loop alla fine di ogni epoca
            for x_batch_val, y_batch_val in val_dataset:
                self._test_step(model, x_batch_val, y_batch_val)
                # val_logits = model(x_batch_val, training=False)
                ## update metriche di validation
                # self.val_acc_metric.update_state(y_batch_val, val_logits)
            val_acc = self.val_acc_metric.result()
            self.val_acc_metric.reset_state()
            logging.info("Validation Acc: %.4f", float(val_acc))
            logging.info("Time Taken:     %.2f s", time.time() - start_time)

    # se le funzioni sono compilate come nodo statico di un grafo di computazione, non le puoi debuggare, pero vanno piu
    # veloce
    @tf.function
    def _train_step(
        self, model: keras.layers.Layer, x: tf.Tensor, y: tf.Tensor
    ) -> tf.Tensor:
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = self.loss_fn(y, logits)
        grads = tape.gradient(loss_value, model.trainable_weights)
        grads_and_vars = zip(grads, model.trainable_weights)
        self.optimizer.apply_gradients(grads_and_vars)
        self.train_acc_metric.update_state(y, logits)
        return loss_value

    @tf.function
    def _test_step(self, model: keras.layers.Layer, x: tf.Tensor, y: tf.Tensor) -> None:
        val_logits = model(x, training=False)
        self.val_acc_metric.update_state(y, val_logits)

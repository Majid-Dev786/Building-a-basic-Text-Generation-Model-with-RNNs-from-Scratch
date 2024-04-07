# First, I need to import TensorFlow, because it's the backbone of our machine learning model.
import tensorflow as tf

# Here's a class where I'll manage the text data. I need it to prepare our data for the model.
class TextData:
    def __init__(self, text):
        # Storing the text and creating a vocabulary out of it. Unique characters matter here.
        self.text = text
        self.vocab = sorted(set(text))
        # Mapping each character to an index and vice versa. It'll help us vectorize the text.
        self.char_to_idx = {char: idx for idx, char in enumerate(self.vocab)}
        self.idx_to_char = {idx: char for idx, char in enumerate(self.vocab)}
        # Turning our text into a list of integers based on our char-to-index mapping.
        self.text_as_int = [self.char_to_idx[char] for char in self.text]

    # Just a method to easily fetch everything I might need later on.
    def get_data(self):
        return self.text_as_int, self.char_to_idx, self.idx_to_char, len(self.vocab)

# This class is where I'll prepare the data for training the model.
class TrainingDataGenerator:
    def __init__(self, text_as_int, seq_length=50):
        # Keeping track of the integer representation of the text and how long each sequence should be.
        self.text_as_int = text_as_int
        self.seq_length = seq_length
        # Calculating the number of examples we'll have based on the text length and sequence length.
        self.examples_per_epoch = len(text_as_int) // (seq_length + 1)
        # Creating a dataset of individual characters for easy sequence creation.
        self.char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
        # Batching these characters into sequences we can feed to the model.
        self.sequences = self.char_dataset.batch(seq_length + 1, drop_remainder=True)

    # A helper method to split sequences into input and target text. It's key for the learning process.
    def split_input_target(self, chunk):
        input_text = chunk[:-1]  # All but the last character
        target_text = chunk[1:]  # All but the first character
        return input_text, target_text

    # Here, I'm putting together the dataset in the form TensorFlow needs to train the model.
    def get_dataset(self, batch_size=64, buffer_size=10000):
        # Mapping the split function to our sequences and then mixing and batching them.
        dataset = self.sequences.map(self.split_input_target)
        dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)
        return dataset

# This class defines our text generation model. It's a simple but effective RNN.
class TextGenerationModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024):
        # Setting up the model layers: Embedding -> LSTM -> Dense
        super(TextGenerationModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.lstm = tf.keras.layers.LSTM(rnn_units, return_sequences=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    # How the data flows through the model
    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.lstm(x)
        output = self.dense(x)
        return output

# I'll use this class to handle the training process.
class ModelTrainer:
    def __init__(self, model, dataset):
        # Storing the model and dataset, and setting up the loss and optimizer.
        self.model = model
        self.dataset = dataset
        self.loss = tf.keras.losses.sparse_categorical_crossentropy
        self.optimizer = tf.keras.optimizers.Adam()

    # The training loop
    def train(self, epochs=30):
        self.model.compile(optimizer=self.optimizer, loss=self.loss)
        for epoch in range(epochs):
            hidden = self.model.reset_states()
            for (batch, (inputs, targets)) in enumerate(self.dataset):
                # Training happens here, with a little bit of magic (i.e., gradient descent).
                with tf.GradientTape() as tape:
                    predictions = self.model(inputs)
                    batch_loss = self.loss(targets, predictions, from_logits=True)
                grads = tape.gradient(batch_loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
                # Printing out the loss every 100 batches to keep track of progress.
                if batch % 100 == 0:
                    print(f"Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy()}")

# This class is our text generator. It uses the trained model to generate new text.
class TextGenerator:
    def __init__(self, model, char_to_idx, idx_to_char):
        # Need the model and the mappings between chars and indices.
        self.model = model
        self.char_to_idx = char_to_idx
        self.idx_to_char = idx_to_char

    # The method to generate text. It's quite the creative part of our journey.
    def generate_text(self, start_string, num_generate=1000):
        # Preparing the input format for the model.
        input_eval = [self.char_to_idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)
        text_generated = []
        self.model.reset_states()

        for _ in range(num_generate):
            # Generating one character at a time and appending it to our generated text.
            predictions = self.model(input_eval)
            predictions = tf.squeeze(predictions, 0)
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
            input_eval = tf.expand_dims([predicted_id], 0)
            text_generated.append(self.idx_to_char[predicted_id])
        return (start_string + ''.join(text_generated))

# And finally, this is where everything comes together.
def main():
    # Creating our text data object with some placeholder text.
    text_data = TextData("""
    Lorem ipsum dolor sit amet, consectetur adipiscing elit. Curabitur varius erat vitae feugiat pulvinar.
    Suspendisse id dapibus justo. Nullam finibus lacus a orci pellentesque vehicula.
    Praesent rhoncus lacus vel metus euismod, vitae aliquam purus finibus.
    """)
    # Getting our data ready for training.
    text_as_int, char_to_idx, idx_to_char, vocab_size = text_data.get_data()
    data_generator = TrainingDataGenerator(text_as_int)
    dataset = data_generator.get_dataset()
    # Building and training the model.
    model = TextGenerationModel(vocab_size)
    trainer = ModelTrainer(model, dataset)
    trainer.train()
    # And here's the fun part: generating text.
    text_generator = TextGenerator(model, char_to_idx, idx_to_char)
    print(text_generator.generate_text("Lorem"))

# Running our main function, which kicks off everything.
if __name__ == "__main__":
    main()

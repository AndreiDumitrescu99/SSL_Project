import matplotlib.pyplot as plt

def plot_loss(history: any, plot_path: str) -> None:

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, label='Training loss')
    plt.plot(epochs, val_loss, label='Validation loss')
    plt.yscale('linear')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()

def plot_accuracy(history: any, plot_path: str) -> None:

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(1, len(accuracy) + 1)

    plt.plot(epochs, accuracy, label='Training accuracy')
    plt.plot(epochs, val_accuracy, label='Validation accuracy')
    plt.yscale('linear')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(plot_path)
    plt.clf()

def translate_prediction(predictions):

  x_pred = []

  for p in predictions:
    maxim = max(p)
    vec = []

    for val in p:
      if val != maxim:
        vec.append(0)
      else:
        vec.append(1)

    x_pred.append(vec)

  return x_pred  
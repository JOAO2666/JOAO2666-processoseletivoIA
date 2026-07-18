import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Classificação MNIST
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o dataset MNIST via tf.keras.datasets.mnist
#   2. Normalizar as imagens para [0, 1] e ajustar o shape para (28, 28, 1)
#   3. Separar um conjunto de validação (ex: validation_split ou split manual)
#   4. Construir uma CNN com 3-4 blocos Conv2D + BatchNormalization + MaxPooling2D,
#      seguida de Dropout antes da camada de saída (10 classes, softmax)
#   5. Treinar com EarlyStopping monitorando a perda de validação
#   6. Exibir a acurácia de validação final no terminal
#   7. Salvar o modelo treinado como "model.h5"
# ---------------------------------------------------------------------------

def main():
    print("Carregando o dataset MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    print("Normalizando e ajustando o shape...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]
    
    print("Construindo a CNN...")
    model = keras.Sequential([
        keras.Input(shape=(28, 28, 1)),
        # Bloco 1
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 2
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Bloco 3
        layers.Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    print("Iniciando treinamento com EarlyStopping...")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    )
    
    # 20% validation split, up to 15 epochs
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=15,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=1
    )
    
    val_accuracy = history.history['val_accuracy'][-1]
    # Se restore_best_weights=True, pegar a acurácia da melhor época
    best_epoch = early_stopping.best_epoch if early_stopping.best_epoch is not None else len(history.history['val_accuracy']) - 1
    best_val_accuracy = history.history['val_accuracy'][best_epoch]
    print(f"\nTreinamento concluído!")
    print(f"Acurácia de validação final (melhor época): {best_val_accuracy:.4f}")
    
    # Evaluate on test for good measure (not strictly required but good practice)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Acurácia no conjunto de teste: {test_acc:.4f}")
    
    model_path = os.path.join(os.path.dirname(__file__), "model.h5")
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")

if __name__ == "__main__":
    main()

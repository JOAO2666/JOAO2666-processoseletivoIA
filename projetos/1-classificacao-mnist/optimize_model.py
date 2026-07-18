import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.h5")
    
    print(f"Carregando o modelo salvo de {model_path}...")
    model = tf.keras.models.load_model(model_path)
    
    print("Convertendo para TensorFlow Lite com otimização (Dynamic Range Quantization)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()
    
    tflite_path = os.path.join(script_dir, "model.tflite")
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
        
    print(f"Modelo otimizado salvo em: {tflite_path}")

if __name__ == "__main__":
    main()

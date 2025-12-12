import { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "./App.css";

const FRUIT_CLASSES = ["Manzana", "Plátano", "Naranja"];
const IMAGE_SIZE = 128;

interface Prediction {
  fruit: string;
  confidence: number;
}

function App() {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [webcamActive, setWebcamActive] = useState(false);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);

  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const intervalRef = useRef<number | null>(null);

  // Cargar el modelo al iniciar
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);

        // Configurar backend de TensorFlow.js
        await tf.ready();

        // Cargar con opciones de compatibilidad para Keras 3.x
        const loadedModel = await tf.loadLayersModel("/model/model.json", {
          strict: false,
          requestInit: {
            credentials: "same-origin",
          },
        });

        // Compilar el modelo
        loadedModel.compile({
          optimizer: "adam",
          loss: "categoricalCrossentropy",
          metrics: ["accuracy"],
        });

        setModel(loadedModel);
        console.log("Modelo cargado exitosamente");
        console.log("Shape de entrada esperado:", loadedModel.inputs[0].shape);
      } catch (error) {
        console.error("Error al cargar el modelo:", error);
        alert(
          "Error al cargar el modelo. Verifica que los archivos estén en /public/model/"
        );
      } finally {
        setIsModelLoading(false);
      }
    };
    loadModel();
  }, []);

  // Preprocesar imagen
  const preprocessImage = async (
    imageElement: HTMLImageElement | HTMLVideoElement
  ): Promise<tf.Tensor> => {
    return tf.tidy(() => {
      // Convertir la imagen a tensor
      let tensor = tf.browser.fromPixels(imageElement);

      // Redimensionar a 128x128
      tensor = tf.image.resizeBilinear(tensor, [IMAGE_SIZE, IMAGE_SIZE]);

      // Normalizar valores de píxeles a [0, 1]
      tensor = tensor.div(255.0);

      // Agregar dimensión de batch
      return tensor.expandDims(0);
    });
  };

  // Realizar predicción
  const predictImage = async (
    imageElement: HTMLImageElement | HTMLVideoElement
  ) => {
    if (!model) return;

    try {
      const preprocessed = await preprocessImage(imageElement);
      const predictions = model.predict(preprocessed) as tf.Tensor;
      const probabilities = await predictions.data();

      // Encontrar la clase con mayor probabilidad
      const maxIndex = probabilities.indexOf(
        Math.max(...Array.from(probabilities))
      );
      const confidence = probabilities[maxIndex] * 100;

      setPrediction({
        fruit: FRUIT_CLASSES[maxIndex],
        confidence: confidence,
      });

      // Limpiar tensores
      preprocessed.dispose();
      predictions.dispose();
    } catch (error) {
      console.error("Error en la predicción:", error);
    }
  };

  // Iniciar webcam
  const startWebcam = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        setWebcamActive(true);

        // Capturar y predecir cada 1 segundo
        intervalRef.current = window.setInterval(() => {
          if (videoRef.current && model) {
            predictImage(videoRef.current);
          }
        }, 1000);
      }
    } catch (error) {
      console.error("Error al acceder a la webcam:", error);
      alert("No se pudo acceder a la cámara. Verifica los permisos.");
    }
  };

  // Detener webcam
  const stopWebcam = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    setWebcamActive(false);
    setPrediction(null);
  };

  // Limpiar al desmontar
  useEffect(() => {
    return () => {
      stopWebcam();
    };
  }, []);

  // Manejar carga de imagen manual
  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.onload = async () => {
        setImagePreview(event.target?.result as string);
        await predictImage(img);
      };
      img.src = event.target?.result as string;
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="app-container">
      <h1>🍎 Clasificador de Frutas</h1>
      <p className="subtitle">Identifica Manzanas, Plátanos y Naranjas</p>

      {isModelLoading && (
        <div className="loading">
          <p>Cargando modelo de IA...</p>
        </div>
      )}

      {!isModelLoading && model && (
        <>
          {/* Sección de Webcam */}
          <div className="section">
            <h2>📹 Captura con Webcam</h2>
            <div className="webcam-container">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                style={{ display: webcamActive ? "block" : "none" }}
              />
              {!webcamActive && (
                <div className="webcam-placeholder">
                  <p>Webcam desactivada</p>
                </div>
              )}
            </div>
            <div className="buttons">
              {!webcamActive ? (
                <button onClick={startWebcam} className="btn-primary">
                  Activar Webcam
                </button>
              ) : (
                <button onClick={stopWebcam} className="btn-danger">
                  Detener Webcam
                </button>
              )}
            </div>
          </div>

          {/* Sección de carga manual */}
          <div className="section">
            <h2>📁 Subir Imagen Manualmente</h2>
            <input
              type="file"
              ref={fileInputRef}
              accept="image/*"
              onChange={handleImageUpload}
              style={{ display: "none" }}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              className="btn-secondary"
            >
              Seleccionar Imagen
            </button>

            {imagePreview && (
              <div className="image-preview">
                <img src={imagePreview} alt="Preview" />
              </div>
            )}
          </div>

          {/* Resultado de la predicción */}
          {prediction && (
            <div className="prediction-result">
              <h2>🎯 Resultado</h2>
              <div className="result-card">
                <p className="fruit-name">{prediction.fruit}</p>
                <p className="confidence">
                  Confianza: {prediction.confidence.toFixed(2)}%
                </p>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${prediction.confidence}%` }}
                  />
                </div>
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;

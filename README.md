# Detector y analizador de ajedrez con YOLO

Proyecto universitario para detectar piezas sobre un tablero real, deducir el movimiento realizado y consultar la mejor jugada con Stockfish. El pipeline acepta frames de una ESP32-CAM, webcam o una imagen local.

## Estructura del dataset

Coloca los archivos YOLO cuando hayas creado el dataset:

```text
dataset/
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
```

Cada etiqueta usa una linea por pieza:

```text
<clase 0-11> <x_centro> <y_centro> <ancho> <alto>
```

Las coordenadas deben estar normalizadas entre 0 y 1. Los nombres y el orden de las clases están definidos en `data.yaml`.

## Preparar el entorno en Windows PowerShell

Desde la carpeta del proyecto:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Se recomienda Python 3.12 por la compatibilidad actual de Ultralytics, PyTorch
y las ruedas CUDA utilizadas por este proyecto.

Si falta la ruta `.\venv\Scripts\Activate.ps1`, borra el entorno creado por
otro terminal o MSYS2 y créalo nuevamente con el comando anterior.

## Instalar PyTorch con CUDA

Instala PyTorch primero con soporte CUDA, por ejemplo CUDA 12.8:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Para CUDA 12.6 usa en su lugar:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Después instala las dependencias restantes:

```powershell
python -m pip install -r requirements.txt
```

Verifica GPU, versión de PyTorch y CUDA:

```powershell
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA de PyTorch:', torch.version.cuda); print('GPU disponible:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

La comprobación correcta debe mostrar:

```text
GPU disponible: True
GPU: NVIDIA GeForce RTX 3050 ...
```

No ejecutes el entrenamiento mientras falte el dataset.

El dataset incluido en esta carpeta es solo una estructura temporal. Puedes reemplazarlo
después sin modificar el código, proporcionando otro directorio y otro `data.yaml`:

```powershell
$env:DATASET_DIR="C:\datasets\ajedrez_nuevo"
$env:DATA_YAML="C:\datasets\ajedrez_nuevo\data.yaml"
python src/train.py --dataset $env:DATASET_DIR --data $env:DATA_YAML
```

## Configuracion

Edita las constantes o variables de entorno en `src/config.py`. Los ajustes principales son:

- `BASE_MODEL`: modelo base, por defecto `yolov8n.pt`.
- `IMGSZ`, `BATCH_SIZE`, `EPOCHS`, `CONF_THRESHOLD`.
- `ESP32_CAPTURE_URL`.
- `STOCKFISH_PATH`.
- `CORNER_JSON` para reutilizar las esquinas seleccionadas.

Ejemplo de variables temporales:

```powershell
$env:ESP32_CAM_IP="192.168.1.50"
$env:STOCKFISH_PATH="C:\tools\stockfish.exe"
$env:YOLO_BATCH_SIZE="4"
```

## Guia paso a paso para probar el proyecto

Las pruebas deben hacerse en este orden para localizar errores facilmente.

### 1. Comprobar la activacion del entorno

Desde la carpeta del proyecto:

```powershell
.\venv\Scripts\Activate.ps1
python --version
python -c "import sys; print(sys.executable)"
```

La segunda orden debe mostrar una ruta dentro de
`C:\Users\PREDATOR\ajedrez_yolo\venv\Scripts`.

### 2. Comprobar las dependencias

```powershell
python -c "import torch, ultralytics, cv2, chess, requests, numpy; print('Dependencias OK')"
python -m compileall src
```

Si falta un modulo, instala nuevamente las dependencias:

```powershell
python -m pip install -r requirements.txt
```

### 3. Comprobar CUDA

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Si `nvidia-smi` muestra la RTX 3050 pero PyTorch imprime `False`, reinstala
exclusivamente PyTorch con CUDA:

```powershell
python -m pip uninstall -y torch torchvision
python -m pip cache purge
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Usa cu126 solamente si cu128 indica que no existe una rueda compatible:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 4. Probar cada script antes del entrenamiento

Ayuda de los comandos:

```powershell
python src/train.py --help
python src/evaluate.py --help
python src/detect.py --help
python src/esp32.py --help
python src/main.py --help
```

Deduccion de una jugada:

```powershell
python -c "import chess; from src.moves import deducir_jugada; inicio=chess.STARTING_FEN; tablero=chess.Board(); tablero.push_uci('e2e4'); print(deducir_jugada(inicio, tablero.fen()))"
```

Debe aparecer `e2e4` y SAN `e4`.

### 5. Probar la captura

Webcam:

```powershell
python src/esp32.py --source webcam --webcam 0
```

Imagen local:

```powershell
python src/esp32.py --source image --image "C:\ruta\de\la\imagen.jpg"
```

ESP32-CAM. Sustituye la IP por la asignada a tu ESP32:

```powershell
python src/esp32.py --source esp32 --url http://192.168.1.50/capture
```

Presiona cualquier tecla en la ventana para cerrar la prueba.

### 6. Preparar y verificar el dataset

Coloca archivos cuando el dataset este listo:

```text
dataset/images/train/
dataset/images/val/
dataset/images/test/
dataset/labels/train/
dataset/labels/val/
dataset/labels/test/
```

Cada imagen debe tener una etiqueta `.txt` con el mismo nombre de archivo.
Ejemplo:

```text
dataset/images/train/imagen001.jpg
dataset/labels/train/imagen001.txt
```

Verifica que existan al menos una imagen y una etiqueta de entrenamiento y
validacion:

```powershell
(Get-ChildItem dataset\images\train\*.*).Count
(Get-ChildItem dataset\labels\train\*.txt).Count
(Get-ChildItem dataset\images\val\*.*).Count
(Get-ChildItem dataset\labels\val\*.txt).Count
```

No debes iniciar el entrenamiento hasta que ambos pares tengan contenido.

## Entrenar

Ejecuta el entrenamiento en dos fases:

```powershell
python src/train.py --epochs 60 --batch 4 --imgsz 640
```

La fase 1 congela las primeras 10 capas durante 30 epocas. La fase 2 descongela todo el modelo y utiliza una tasa de aprendizaje menor.

## Evaluar

Evaluacion sobre val:

```powershell
python src/evaluate.py --model models/best.pt --split val --batch 4 --imgsz 640
```

Evaluacion sobre test:

```powershell
python src/evaluate.py --model models/best.pt --split test --batch 4 --imgsz 640
```

## Probar una deteccion

```powershell
python src/detect.py ruta\imagen.jpg --model models/best.pt --conf 0.40 --save salida.jpg
```

## Probar la captura

ESP32-CAM:

```powershell
python src/esp32.py --source esp32 --url http://192.168.1.50/capture
```

Webcam:

```powershell
python src/esp32.py --source webcam --webcam 0
```

Imagen local:

```powershell
python src/esp32.py --source image --image ruta\imagen.jpg
```

## Ejecutar el pipeline completo

ESP32-CAM:

```powershell
python src/main.py --source esp32 --model models/best.pt
```

Webcam:

```powershell
python src/main.py --source webcam --webcam 0 --model models/best.pt
```

Imagen local:

```powershell
python src/main.py --source image --image ruta\imagen.jpg --model models/best.pt
```

La primera vez puede activarse la seleccion manual si no se detectan las esquinas. Haz clic en orden: superior izquierda, superior derecha, inferior derecha e inferior izquierda. Presiona `Enter` para guardar las esquinas en `board_corners.json`.

Uso:

- `espacio`: confirma el FEN visible y consulta la jugada de Stockfish.
- `q`: salir.

Stockfish debe estar instalado por separado y su ruta debe configurarse mediante `STOCKFISH_PATH`.

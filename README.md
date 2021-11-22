# Dicom Classifier

Dicom Classifier es una libreria para manejar imagenes médicas dicom y con diferentes técnicas de clasificación.

## Dependencias
La libreria usa:
- **libdcmtk-dev**: para el manejo de imágenes dicom.
- **libeigen3-dev**: operaciones matemáticas que.

Para instalarlos puedes ejecutar el script `checker.sh`

```bash
chmod +x ./checker.sh
./checker.sh
```

Tambien se pueden instalar de manera manual:
```bash
sudo apt install libeigen3-dev libdcmtk-dev
```

## Instalación

Antes de usar la libreria se necesita realizar el proceso de compilación para generar los archivos necesarios e importarlos directamente en su proyecto.

Para compilar se utiliza QT creator.

### QT

1. Primero se clona el proyecto en su computadora.
2. Abrir el archivo .pro con QT Creator.
3. Configurar la carpeta donde se compilará
4. Compilar el proyecto

### Makefiles (proximamente)

## Uso

Para usar esta libreria asegurese de haber compilado este proyecto y haber generado los archivos `.o` y `.a`.

La libreria se ha usado con los IDEs QT Creator y CodeBlocks.

### QT
Para usarlo con QT inicialice un nuevo proyecto y asegurece que su archivo de configuracion `.pro` tenga lo siguiente:

```
INCLUDEPATH += /dicom-classifier/DicomClassifier # Aqui es el path del proyecto con las cabeceras.

LIBS += -L/dicom-classifier/build   # Directorio con los archivos compilados
    -lDicomClassifier \
    -ldcmdata \
    -ldcmimgle \
    -ldcmimage \
    -ldcmjpeg

```

### Codeblocks

Para usarlo en codeblocks inicie un nuevo proyecto y asegurece de ejecutar el script `checker.sh`.

Dirigirse a configuraciones y compilador, en la pestaña __linker settings__ agregue el archivo `libDicomClasifier.a` que se encuentra en la carpeta __build__ despues del proceso de compilacion. Tambien incluya las siguientes opciones:

```
-lDicomClassifier
-ldcmdata
-ldcmimgle
-ldcmimage
-ldcmjpeg
```

Ahora en la pestaña `Search directories > compiler` agregue el directorio donde se encuentran los archivos `.h` del proyecto.

```
/home/user/projects/dicom-classifier/DicomClassifier
```

En la pestaña `Search directories > linker` agregue el directorio donde se se encuentran los archivos compilados.

```
/home/user/projects/dicom-classifier/build
```

## Ejemplos

Puede encontrar ejemplos de uso [aquí](examples)

## Autores

- [Wilson Aguilar](https://www.github.com/waguilars)
- [Gabriel Cacuango]()

****************************************************************
*			Atril inteligente		       *
*		Entrenamiento de las redes neuronales	       *
****************************************************************	

Esta carpeta contiene los archivos necesarios para poder llevar a cabo entrenamientos con redes neuronales

1º Para poder realizar el entrenamiento primero hay que procesar los datos.
	Necesario instalar los paquetes Librosa y PrettyMidi
	1.1 El dataset utilizado ha sido MAPS, descargable en: "http://www.tsi.telecom-paristech.fr/aao/en/2010/07/08/maps-database-a-piano-database-for-multipitch-estimation-and-automatic-transcription-of-music/
	1.2 Los archivos deben estar separados en carpetas llamadas Isolated(archivos aislados), Chords(archivos de acordes) y MusicalPieces(Archivos de piezas musicales)
	1.3 La ruta hasta el dataset en este proyecto es: '../dataset/MusicalPieces', esto puede encontrarse en "CargarGuardar.py", función loadFiles()
	1.4 Una vez solucionado los subpuntos anteriores, ejecutar CQT.py

2º Entrenar perceptrón multicapa
	Necesario instalar el paquete Keras
	2.1 Abrir archivo Perceptron.py y ejecutar. Los modelos se guardarán en la carpeta modelos, y se guardará modelo, pesos, valores de normalización, gráficas F1 y error total y algunas imagenes de prueba.

3º Entrenar red convolucional
	Necesario instalar el paquete Keras
	3.1 Abrir archivo Convolucional.py y ejecutar. Los modelos se guardarán en la carpeta modelos, y se guardará modelo, pesos, valores de normalización, gráficas F1 y error total y algunas imagenes de prueba.


Notas:
	-Carpeta archivosPrueba contiene los archivos en CQT para realizar pruebas postentrenamiento
	-datasetProcesado contiene los archivos necesarios para entrenar los modelos
	-imagenes: Contiene algunas imagenes CQT obtenidas a partir de la conversión de los archivos del dataset
	-modelos: Contiene los modelos entrenados y algunos entrenados previamente por el autor
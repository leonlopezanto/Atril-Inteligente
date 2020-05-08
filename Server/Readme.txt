****************************************************************
*			Atril inteligente		       *
*			   Servidor web	       		       *
****************************************************************	

Esta carpeta contiene los archivos necesarios para poder poner en marcha el servidor y utilizar el servicio en un navegador web

¡¡¡IMPORTANTE!!!:
	-Este desarrollo contiene llamadas al sistema para convertir archivos MusicXML y Midi con la ayuda de MuseScore3.
	-Es necesario tener instalado MuseScore3 y añadirlo a las variables de entorno para poder realizar las llamadas al sistema.
	(Añadir variables de entorno: Panel de control ->Configuración avanzada del sistema -> Variables de Entorno -> Variables del sistema -> Editar variable del sistema -> Especificar variable de entorno PATH añadiendo la ruta del ejecutable (Pulsar examinar))

1º Para poder utilizar el servidio hay que instalar el servidor
	Necesario instalar los paquetes Flask, Librosa, keras y PrettyMidi
	1.1 Abrimos Server.py y ejecutamos.
	1.2 Accedemos a Localhost en un navegador web
	1.3 Introducimos archivos de audio y partitura

Notas:
	-Carpeta NeuralNetwork contiene el modelo los pesos y los valores de normalización. En el archivo Alineamiento.py tenemos que indicar la arquitectura (linea 106).
	-La carpeta static contiene los archivos Javascript necesarios
	-La carpeta templates contiene los archivos html
	-La carpeta uploads contiene los archivos utilizados por el servicio en el tiempo. 
	-La carpeta EJEMPLOS contiene el archivo de audio y partitura de Rivers Flows in You e Italian Polka, por si desearas realizar una prueba.	

	-El archivo dtw.py es el algoritmo DTW obtenido del repositorio: "https://github.com/craffel/djitw/blob/master/djitw/dtw.py"
	-mpostool.py obtiene los datos de los archivos MPOS necesarios para alinear.
	-El visor de partituras en el navegador ha sido desarrollado por VimFree: https://wim.vree.org/js/abcweb.html
	(Gracias a VimFree y a DJITW)
	

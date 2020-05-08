
"""
Servidor
"""

#Medir tiempo
from time import time


from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os

#Propias
import Alineamiento

#Instancia objeto Flash
app = Flask(__name__)

#Carpeta de subida
UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)
cors = CORS(app)



@app.route("/", methods=['POST', 'GET'])
def upload():
    print(request.method)
    print(request.files)
    if request.method == 'POST' and request.files:
        print("Comienzo")
        #Obtenemos los archivos
        xml = request.files['xmlFile']
        audio = request.files['audioFile']

        #Validamos y procesamos
        timeiniVerificar = time()
        pathAudio, pathNuevoMidi, pathNuevoMpos, isMidi = procesarArchivos(xml, audio)
        timefinVerificar = time()
        timeVefTotal = timefinVerificar - timeiniVerificar
        print("Tiempo verificación: ", timeVefTotal)

        if (pathAudio == False):
            return '<h1>Problema con el archivo de audio</h1>'
        elif (pathNuevoMidi == False):
            return '<h1>Problema con el archivo de partitura</h1>'
        else:
            print('Archivos subidos con éxito')

        #Obtenemos los tiempos de inicio de los compases
        timeiniAl = time()
        iniCompas = Alineamiento.Alinear(pathAudio, pathNuevoMidi, pathNuevoMpos)
        timefinAl = time()
        timeAlTotal = timefinAl - timeiniAl
        print("Tiempo alineacion total: ", timefinAl - timeiniAl)

        print("Tiempo total de ejecución: ", timeAlTotal + timeVefTotal)


        if isMidi:
            scoreFile, extScore = os.path.splitext(secure_filename(xml.filename.replace(" ","")))
            filexml = secure_filename(scoreFile+'.musicxml')
        else:
            filexml = secure_filename(xml.filename.replace(" ",""))
        fileaudio = secure_filename(audio.filename.replace(" ",""))

        #devuelve nombre de plantilla, nombre xml, nombre audio y array de tiempos de compás
        return render_template('alineado.html', xml=filexml, audio=fileaudio, iniCompas=iniCompas)

    else:
        return render_template('upload.html')


@app.route('/uploads/<filename>', methods=['GET','POST'])
def send_file(filename):
    print('Enviando archivos')
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename, as_attachment=True)


def procesarArchivos(score, audio):
    '''
    :param score: Archivo XML, MusicXML o Midi para procesar
    :param audio: Archivo de audio para procesar
    :return: la ruta a archivos necesarios para alinear
    '''

    # print('Procesando archivos')


    # validacion archivo xml, musicxml, midi

    _, extScore = os.path.splitext(secure_filename(score.filename))
    if extScore not in [".xml", ".musicxml", ".mid"]:
        return None, False
    else:
        print("Partitura subida")
        scoreFile = score.filename.replace(" ", "")
        score.save(os.path.join(UPLOAD_FOLDER, secure_filename(scoreFile)))
        scoreFile, _ = os.path.splitext(scoreFile)
        # print("Archivo XML subido: ", xml.filename)

    # validacion archivo audio
    _, extAudio = os.path.splitext(secure_filename(audio.filename))
    if extAudio not in [".mp3", ".wav"]:
        return False, None
    else:
        print("Audio subido")
        audioFile = audio.filename.replace(" ", "")
        audio.save(os.path.join(UPLOAD_FOLDER, secure_filename(audioFile)))
        audioFile, _ = os.path.splitext(audioFile)
        # print("Archivo Audio subido: ", audio.filename)

    pathScore = os.path.join(UPLOAD_FOLDER,  scoreFile + extScore)
    pathAudio = os.path.join(UPLOAD_FOLDER, audioFile + extAudio)

    #Musescore debe estar añadido al PATH del sistema
    # Creamos el objeto midi (si es necesario) y MPOS con la siguiente sentencia:
    if extScore != ".mid":
        pathMidi = os.path.join(UPLOAD_FOLDER, scoreFile + '.mid')
        if (not (os.path.isfile(pathMidi))):
            sentenciaSysMidi = 'MuseScore3 "' + pathScore + '" -o "' + pathMidi + '"'
            print("Creando Midi")
            os.system(sentenciaSysMidi)
            print("Midi creado")
        isMidi = False
    else:
        pathMidi = os.path.join(UPLOAD_FOLDER, score.filename)
        pathNuevoXML = os.path.join(UPLOAD_FOLDER, scoreFile + '.musicxml')
        if (not (os.path.isfile(pathNuevoXML))):
            sentenciaSysXML = 'MuseScore3 "' + pathScore + '" -o "' + pathNuevoXML + '"'
            print("Creando XML")
            os.system(sentenciaSysXML)
            print("XML Creado")
        isMidi = True

    pathNuevoMpos = os.path.join(UPLOAD_FOLDER, scoreFile + '.mpos')
    print("Creando MPOS")
    sentenciaSysMpos = 'MuseScore3 "' + pathScore + '" -o "' + pathNuevoMpos + '"'
    os.system(sentenciaSysMpos)
    print("MPOS Creado")


    return pathAudio, pathMidi, pathNuevoMpos, isMidi

######################################### MAIN #########################################

#Lanzamiento del servidor
if __name__ == '__main__':
    app.debug = True
    app.run(port=8888)
import xml.etree.ElementTree as ET
import sys
from os import remove



def get_mpos_data(mposFile):
    '''
    Obtiene los inicios de compás de la partitura a partir del MPOS
    :param mposFile: ruta archivo MPOS
    :return: Tiempo de inicio de compás
    '''
    try:
        root = ET.parse(mposFile).getroot()
        f = open(mposFile, 'wb')
        bartimes = list()  # np.zeros((len(root.findall('events/event')),2))
        for type_tag in root.findall('events/event'):
            bartimes.append(float(type_tag.get('position')) / 1000.0)
            # f.write('%.2f%.2f' % (float(type_tag.get('position')) / 1000.0, 1.0+float(type_tag.get('elid'))))
        f.close()
        remove(mposFile)
        return bartimes
    except Exception as e:
        raise type(e)('In func get_mpos_data: ' + str(e)).with_traceback(sys.exc_info()[2])


# ---------------------------------------------------------------------------------------------------------------
# SAMPLE MAIN
# if __name__ == "__main__":
#     mposFile = './temp/hb.mpos'
#     try:
#         bartimes = get_mpos_data(mposFile)
#         print(bartimes)
#     except Exception as e:
#         print('***** ERROR *****')
#         print(e)
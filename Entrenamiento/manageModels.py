



from matplotlib import pyplot as plt
from keras.models import model_from_json
import numpy as np
plt.switch_backend('agg')


def save_scores_in_fig(val_loss_list, train_loss_list, val_scores_list, name):
    plt.suptitle('')
    plt.subplot(2, 1, 1)
    plt.semilogy(np.hstack(val_loss_list), linewidth=2, label='val_loss')  # semilogx would be better...
    plt.semilogy(np.hstack(train_loss_list), linewidth=2, label='train_loss')
#    plt.ylim([0, 1])
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.subplot(2, 1, 2)
    for score_metric in val_scores_list[0].keys():
        plt.plot(np.hstack([elem[score_metric] for elem in val_scores_list]), linewidth=2, label=score_metric)
    plt.ylim([0, 1.2])
    plt.legend(loc='upper left')
    plt.grid(True)
    
    nameDir = 'Model_'+name
    path = './modelos/' + nameDir + '/' + name
    
    plt.savefig(path)
    plt.close()


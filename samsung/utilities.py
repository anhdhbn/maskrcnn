import math
import matplotlib.pyplot as plt

def draw_loss(history, name, t_loss  =  [], val_loss  =  []):
    print(history)
    training_loss = t_loss + history['loss']
    test_loss = val_loss + history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    plt.plot(epoch_count, training_loss, 'r--')
    plt.plot(epoch_count, test_loss, 'b-')
    plt.legend(['Training Loss', 'Val Loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'{ROOT_DIR}logs/{name}.png')
    return training_loss, test_loss

def step_decay(epoch):
    if(epoch <= 5):
        initial_lrate = 0.001
        epochs_drop = 3.0
        return calc_lr(initial_lrate, epochs_drop, epoch)
    elif (epoch <= 10):
        initial_lrate = 0.001
        epochs_drop = 3.0
        return calc_lr(initial_lrate, epochs_drop, epoch)
    else:
        initial_lrate = 0.001 / 10
        epochs_drop = 5
        return calc_lr(initial_lrate, epochs_drop, epoch)

def calc_lr(initial_lrate, epochs_drop, epoch):
    keep = 0.5
    lrate = initial_lrate * math.pow(keep,  
            math.floor((1+epoch)/epochs_drop))
    return lrate
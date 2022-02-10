def print_progress(epoch, total_number_epochs, loss, training=True):
    """
    A function that prints the current epoch and the average loss on a given image during this epoch
    """
    if epoch == total_number_epochs:
        print(f'Step: {epoch + 1}/{total_number_epochs} - {"Training" if training else "Testing"} loss: {loss: .4f}')
    else:
        print(f'Step: {epoch + 1}/{total_number_epochs} - {"Training" if training else "Testing"} loss: {loss: .4f}', end="\r")

if __name__ == "__main__":
    print_progress(1, 4, 0.0567)
    print_progress(2, 4, 0.0378)
    print_progress(4, 4, 0.0019)
    

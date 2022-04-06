import matplotlib.pyplot as plt

def loss_printer(train_loss, test_loss, tol, max_iter, beta, lam, MAX_EPOCH, num_epochs, lr_model, lr_fb):

    plt.figure()
    plt.plot(train_loss[1:], label="train")
    plt.plot(test_loss[1:], 'r', label="test")
    plt.title(f"tol={tol}, max_iter={max_iter}, beta={beta}, lam={lam}, MAX_EPOCH={MAX_EPOCH}, num_epochs={num_epochs}, lr_model={lr_model}, lr_fb={lr_fb}")
    plt.legend()
    plt.semilogy()
    plt.savefig(f"tol={tol}_max_iter={max_iter}_beta={beta}_lam={lam}_MAX_EPOCH={MAX_EPOCH}_num_epochs={num_epochs}_lrmodel={lr_model}_lrfb={lr_fb}.png")
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def plot_loss(train_loss, val_loss, title, save_path=None, skip=0):
    # Assuming model is an instance of MyModel and training is done
    plt.figure(figsize=(10, 5))
    if train_loss is not None:
        plt.plot(train_loss[skip:], label='Training Loss')
    if val_loss is not None:
        plt.plot(val_loss[skip:], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def plot_lr(lr, title, save_path=None):
    plt.figure(figsize=(10, 5))
    plt.plot(lr, label='Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title(title)
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def do_graphing_train(metrics_callback, log_dir, skip=150):
    plot_loss(metrics_callback.train_losses, metrics_callback.val_losses, "plot combined loss",
              save_path=os.path.join(log_dir, "noskip_loss.pdf"), skip=0)
    plot_loss(metrics_callback.train_losses, metrics_callback.val_losses, "plot combined loss",
              save_path=os.path.join(log_dir, "skip_loss.pdf"), skip=skip)
    # only plotting train loss
    plot_loss(metrics_callback.train_losses, val_loss=None, title="train combined loss",
              save_path=os.path.join(log_dir, "noskip_train_loss.pdf"), skip=0)
    plot_loss(metrics_callback.train_losses, val_loss=None, title="train combined loss",
              save_path=os.path.join(log_dir, "skip_train_loss.pdf"), skip=skip)

    plot_loss(metrics_callback.train_lap, val_loss=None,
              title="train lap loss",
              save_path=os.path.join(log_dir, "noskip_train_lap_loss.pdf"), skip=0)
    plot_loss(metrics_callback.train_lap, val_loss=None,
              title="train lap loss",
              save_path=os.path.join(log_dir, "skip_train_lap_loss.pdf"), skip=skip)

    plot_loss(metrics_callback.train_lap, val_loss=metrics_callback.val_lap,
              title=" lap loss",
              save_path=os.path.join(log_dir, "noskip_lap_loss.pdf"), skip=0)

    plot_loss(metrics_callback.train_lap, val_loss=metrics_callback.val_lap,
              title="lap loss",
              save_path=os.path.join(log_dir, "skip_lap_loss.pdf"), skip=skip)

    plot_loss(metrics_callback.train_contrastive_loss, val_loss=metrics_callback.val_contrastive_loss,
              title="contrastive loss",
              save_path=os.path.join(log_dir, "contrastive_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_contrastive_loss, val_loss=metrics_callback.val_contrastive_loss,
              title="contrastive loss skip",
              save_path=os.path.join(log_dir, "contrastive_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_contrastive_loss, val_loss=None, title="train contrastive loss",
              save_path=os.path.join(log_dir, "train_contrastive_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_contrastive_loss, val_loss=None, title="train contrastive loss skip",
              save_path=os.path.join(log_dir, "train_contrastive_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_similarsp_loss, val_loss=metrics_callback.val_similarsp_loss,
              title="similar sp loss",
              save_path=os.path.join(log_dir, "similarsp_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_similarsp_loss, val_loss=metrics_callback.val_similarsp_loss,
              title="similar sp loss skip",
              save_path=os.path.join(log_dir, "similarsp_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_similarsp_loss, val_loss=None, title="train similar sp loss",
              save_path=os.path.join(log_dir, "train_similarsp_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_similarsp_loss, val_loss=None, title="train similar sp loss skip",
              save_path=os.path.join(log_dir, "train_similarsp_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_impurity_loss, val_loss=metrics_callback.val_impurity_loss,
              title="impurity loss",
              save_path=os.path.join(log_dir, "impurity_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_impurity_loss, val_loss=metrics_callback.val_impurity_loss,
              title="impurity loss skip",
              save_path=os.path.join(log_dir, "impurity_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_impurity_loss, val_loss=None, title="train impurity loss",
              save_path=os.path.join(log_dir, "train_impurity_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_impurity_loss, val_loss=None, title="train impurity loss skip",
              save_path=os.path.join(log_dir, "train_impurity_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_mean_asa, val_loss=metrics_callback.val_mean_asa,
              title="asa ",
              save_path=os.path.join(log_dir, "asa_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_mean_asa, val_loss=metrics_callback.val_mean_asa,
              title="asa skip",
              save_path=os.path.join(log_dir, "asa_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_mean_asa, val_loss=None, title="train asa",
              save_path=os.path.join(log_dir, "train_asa_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_mean_asa, val_loss=None, title="train asa skip",
              save_path=os.path.join(log_dir, "train_asa_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_pos_loss, val_loss=metrics_callback.val_pos_loss,
              title="pos loss",
              save_path=os.path.join(log_dir, "pos_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_pos_loss, val_loss=metrics_callback.val_pos_loss,
              title="pos loss skip",
              save_path=os.path.join(log_dir, "pos_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_pos_loss, val_loss=None, title="train pos loss",
              save_path=os.path.join(log_dir, "train_pos_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_pos_loss, val_loss=None, title="train pos loss skip",
              save_path=os.path.join(log_dir, "train_pos_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_recon_loss, val_loss=metrics_callback.val_recon_loss,
              title="recon loss",
              save_path=os.path.join(log_dir, "recon_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_recon_loss, val_loss=metrics_callback.val_recon_loss,
              title="recon loss skip",
              save_path=os.path.join(log_dir, "recon_loss_skip.pdf"), skip=skip)

    plot_loss(metrics_callback.train_recon_loss, val_loss=None, title="train recon loss",
              save_path=os.path.join(log_dir, "train_recon_loss_noskip.pdf"), skip=0)
    plot_loss(metrics_callback.train_recon_loss, val_loss=None, title="train recon loss skip",
              save_path=os.path.join(log_dir, "train_recon_loss_skip.pdf"), skip=skip)

    ##### Val enforce connecitivity only metrics

    plot_loss(train_loss=None, val_loss=metrics_callback.val_asa_enforce,
              title="asa enforce val only",
              save_path=os.path.join(log_dir, "asa_noskip_every25_enforce.pdf"), skip=0)

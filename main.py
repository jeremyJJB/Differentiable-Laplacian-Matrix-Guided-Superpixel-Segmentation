import os
import lightning as pl
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from MyModules import AISuperpixModel
from datasets.BSD import CustomDataModule
import torch
from utils.graphing import plot_lr, do_graphing_train
from lightning.pytorch import seed_everything
from utils.util import MetricsCallback, PrintLossesCallback, ValEpochGraphs #, TrainEpochGraphs
from datasets.BSD import make_train_transform, make_val_transform
import yaml
from albumentations import RandomCrop
from eval_spixel.infer import rename_checkpoint_keys


def train_val_sep(hp_config=None):
    # train and val separate
    train_transform = make_train_transform(hp_config['train_size'], model_name=hp_config['model_name'])
    has_random_crop = any(isinstance(t, RandomCrop) for t in train_transform.transforms)
    assert has_random_crop, "the train data should have random crop"
    val_transform = make_val_transform(hp_config['val_size'],model_name=hp_config['model_name'])

    model = AISuperpixModel(learning_rate=hp_config['lr'],
                          model_name=hp_config['model_name'], margin=1.5,
                          random_sample_size=100, device=hp_config['device'],
                          loss_weights=hp_config['loss_weights'],
                          train_height=hp_config['train_size'],
                          train_width=hp_config['train_size'],
                          val_height=hp_config['val_size'],
                          val_width=hp_config['val_size'],
                          batch_size=hp_config['batch_size'],
                          stride=hp_config['stride'],
                          wd=hp_config['weight_decay'],
                          loss_name=hp_config['loss_name'],
                          lr_decay_epoch=hp_config['LR_decay_epoch'],)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_recon_loss",
        mode="min",
        save_last=True,
        filename="best-loss-{epoch}-{val_recon_loss:.4f}",
        save_weights_only=True,
        enable_version_counter=True,
    )

    lr_monitor = LearningRateMonitor("epoch")
    metrics_callback = MetricsCallback(train_type='Train', model_name=hp_config['model_name'])
    # train_viz = TrainEpochGraphs(num_imgs=2, run_type='Train', num_sp=3, total_sp=(hp_config['train_size']//hp_config['stride']) * (hp_config['train_size']//hp_config['stride']), model_name=hp_config['model_name'])
    val_viz = ValEpochGraphs(num_imgs=8, run_type='Val', num_sp=1,
                                     total_sp=(hp_config['val_size'] // hp_config['stride']) * (
                                                 hp_config['val_size'] // hp_config['stride']), model_name=hp_config['model_name'])
    trainer = pl.Trainer(
        default_root_dir=hp_config['result_path'],
        accelerator=hp_config['accelerator'],
        devices="auto",
        max_epochs=hp_config['epochs'],
        enable_progress_bar=False,
        callbacks=[
            checkpoint_callback,
            metrics_callback,
            # train_viz,
            val_viz,
            PrintLossesCallback(train_type='Train'),
            lr_monitor,
        ],
        limit_train_batches=hp_config['num_batches'],
        limit_val_batches=hp_config['num_batches'],
    )


    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if not os.path.exists(trainer.log_dir):
        os.makedirs(trainer.log_dir)
    # Save to YAML
    with open(os.path.join(trainer.log_dir, "config.yaml"), "w") as f:
        yaml.dump(hp_config, f, default_flow_style=False)


    data_module_bsd = CustomDataModule(dataset_root=hp_config['data_dir'],
                                       batch_size=hp_config['batch_size'],
                                       num_workers=hp_config['num_workers'],
                                       train_transform=train_transform, eval_transform=val_transform,
                                       merge_train_val=False, train_size=hp_config['train_size'],val_size=hp_config['val_size'],model_name=hp_config['model_name'],)
    trainer.fit(model, datamodule=data_module_bsd)

    # Get best epoch index
    print(trainer.checkpoint_callback.best_model_path)
    best_epoch = int(trainer.checkpoint_callback.best_model_path.split("epoch=")[-1].split("-")[0]) + 1
    print(f"Lowest validation loss occurred at epoch: {best_epoch}")

    path_txt_files = trainer.log_dir + "/txt/"
    if not os.path.exists(path_txt_files):
        os.makedirs(path_txt_files)

    metrics_callback.write_to_file(dir_path=path_txt_files)


    do_graphing_train(metrics_callback=metrics_callback, log_dir=trainer.log_dir)
    if hp_config['model_name'] == 'cds':
        try:
            plot_lr(lr_monitor.lrs['poly_lr/pg1'], "plot lr model bias", save_path=os.path.join(trainer.log_dir, "lr_bias_model.pdf"))
            plot_lr(lr_monitor.lrs['poly_lr/pg2'], "plot lr model weights", save_path=os.path.join(trainer.log_dir, "lr_weights_model.pdf"))
            plot_lr(lr_monitor.lrs['lr-Adam'], "plot lr MI", save_path=os.path.join(trainer.log_dir, "lr_weights_mi.pdf"))
        except Exception:
            print("falling back to LR plotting")
            plot_lr(lr_monitor.lrs['lr-Adam/pg1'], "plot lr bias",
                    save_path=os.path.join(trainer.log_dir, "lr_bias.pdf"))
            plot_lr(lr_monitor.lrs['lr-Adam/pg2'], "plot lr weights",
                    save_path=os.path.join(trainer.log_dir, "lr_weights.pdf"))
    elif hp_config['model_name'] == 'scn' or hp_config['model_name'] == 'ainet' or hp_config['model_name'] == 'ssm':
        plot_lr(lr_monitor.lrs['lr-Adam/pg1'], "plot lr bias", save_path=os.path.join(trainer.log_dir, "lr_bias.pdf"))
        plot_lr(lr_monitor.lrs['lr-Adam/pg2'], "plot lr weights", save_path=os.path.join(trainer.log_dir, "lr_weights.pdf"))
    else:
        print("The LR was not correct and not plot was made")
    return best_epoch


def train_val_combine(hp_config, max_epochs:int=5):
    # final training before test inference

    train_transform = make_train_transform(hp_config['train_size'],model_name=hp_config['model_name'])
    has_random_crop = any(isinstance(t, RandomCrop) for t in train_transform.transforms)
    assert has_random_crop, "the train data should have random crop"
    val_transform = make_val_transform(hp_config['val_size'],model_name=hp_config['model_name'])
    model = AISuperpixModel(learning_rate=hp_config['lr'],
                          model_name=hp_config['model_name'], margin=1.5,
                          random_sample_size=100, device=hp_config['device'],
                          loss_weights=hp_config['loss_weights'],
                          train_height=hp_config['train_size'],
                          train_width=hp_config['train_size'],
                          val_height=hp_config['val_size'],
                          val_width=hp_config['val_size'],
                          batch_size=hp_config['batch_size'],
                          stride=hp_config['stride'],
                          wd=hp_config['weight_decay'],
                          loss_name=hp_config['loss_name'],
                          lr_decay_epoch=hp_config['LR_decay_epoch'])

    if hp_config['weights_to_load'] == "tv_only":
        # This is used for ssm
        print("doing TV, not weight loading here, weight loading could be in model")
    elif len(hp_config['weights_to_load']) > 0:
        print("we are continue training")
        print(f"loading weights from {hp_config['weights_to_load']}")
        if hp_config['model_name'] == "cds":
            weight_load = torch.load(hp_config['weights_to_load'], map_location=torch.device(hp_config['device']))
            # load weights for main model
            sd = weight_load['state_dict']
            only_model = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
            load_status = model.model.load_state_dict(only_model, strict=True)
            # load the mi weights
            only_mi = {k[len("mi_estimator."):]: v for k, v in sd.items() if k.startswith("mi_estimator.")}
            load_status2 = model.mi_estimator.load_state_dict(only_mi, strict=True)

            # # Check for missing or unexpected keys:
            if load_status.missing_keys or load_status2.missing_keys:
                raise Exception(
                    f"State dict loading mismatch:\n"
                    f"Missing keys: {load_status.missing_keys}\n"
                    f"Unexpected keys: {load_status.unexpected_keys}"
                )
            elif load_status.unexpected_keys or load_status2.unexpected_keys:
                print(f"Unexpected keys: {load_status.unexpected_keys}")
            else:
                print("all keys loaded")

        elif hp_config['model_name'] == "scn" or hp_config['model_name'] == "ainet":
            weight_load = torch.load(hp_config['weights_to_load'], map_location=torch.device(hp_config['device']))
            state_dict = weight_load['state_dict']
            new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
            load_status = model.model.load_state_dict(new_state_dict, strict=False)

            # # Check for missing or unexpected keys:
            if load_status.missing_keys:
                raise Exception(
                    f"State dict loading mismatch:\n"
                    f"Missing keys: {load_status.missing_keys}\n"
                    f"Unexpected keys: {load_status.unexpected_keys}"
                )
            elif load_status.unexpected_keys:
                print(f"Unexpected keys: {load_status.unexpected_keys}")
            else:
                print("all keys loaded")

    else:
        print("no weights to load doing TV from scratch")


    retrain_checkpoint = ModelCheckpoint(
        monitor="val_recon_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        filename="best-loss-{epoch}-{val_recon_loss:.6f}",
        save_weights_only=True,
        enable_version_counter=True,
    )


    metrics_callback_tv = MetricsCallback(train_type='TrainVal', model_name=hp_config['model_name'])
    lr_monitor_tv = LearningRateMonitor("epoch")
    data_module_bsd = CustomDataModule(dataset_root=hp_config['data_dir'],
                                       batch_size=hp_config['batch_size'],
                                       num_workers=hp_config['num_workers'], train_transform=train_transform,
                                       eval_transform=val_transform, merge_train_val=True, train_size=hp_config['train_size'],val_size=hp_config['val_size'],model_name=hp_config['model_name'])
    val_viz_tv = ValEpochGraphs(num_imgs=4, run_type='Val', num_sp=3,
                                     total_sp=(hp_config['val_size'] // hp_config['stride']) * (
                                                 hp_config['val_size'] // hp_config['stride']), model_name=hp_config['model_name'])

    # Retrain from scratch for the same number of epochs as before
    trainer_retrain = pl.Trainer(
        max_epochs=max_epochs,
        enable_checkpointing=True,
        default_root_dir=hp_config['result_path'],
        accelerator=hp_config['accelerator'],
        devices="auto",
        enable_progress_bar=False,
        callbacks=[
            PrintLossesCallback(train_type='TrainVal'),
            lr_monitor_tv,
            val_viz_tv,
            metrics_callback_tv,
            retrain_checkpoint,
        ],
        limit_train_batches = hp_config['num_batches'],
        limit_val_batches=hp_config['num_batches'],

    )

    trainer_retrain.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer_retrain.logger._default_hp_metric = None  # Optional logging argument that we don't need

    if not os.path.exists(trainer_retrain.log_dir):
        os.makedirs(trainer_retrain.log_dir)
    with open(os.path.join(trainer_retrain.log_dir, "config.yaml"), "w") as f:
        yaml.dump(hp_config, f, default_flow_style=False)


    trainer_retrain.fit(model, datamodule=data_module_bsd)
    path_txt_files = trainer_retrain.log_dir + "/txt/"
    if not os.path.exists(path_txt_files):
        os.makedirs(path_txt_files)

    metrics_callback_tv.write_to_file(dir_path=path_txt_files)
    do_graphing_train(metrics_callback=metrics_callback_tv, log_dir=trainer_retrain.log_dir)

    if hp_config['model_name'] == 'cds':
        plot_lr(lr_monitor_tv.lrs['poly_lr/pg1'], "plot lr model bias",
                save_path=os.path.join(trainer_retrain.log_dir, "lr_bias.pdf"))
        plot_lr(lr_monitor_tv.lrs['poly_lr/pg2'], "plot lr model weights",
                save_path=os.path.join(trainer_retrain.log_dir, "lr_weights.pdf"))
        plot_lr(lr_monitor_tv.lrs['lr-Adam'], "plot lr MI", save_path=os.path.join(trainer_retrain.log_dir, "lr_weights.pdf"))
    elif hp_config['model_name'] == 'scn' or hp_config['model_name'] == 'ainet' or hp_config['model_name'] == 'ssm':
        plot_lr(lr_monitor_tv.lrs['lr-Adam/pg1'], "plot lr bias", save_path=os.path.join(trainer_retrain.log_dir, "lr_bias.pdf"))
        plot_lr(lr_monitor_tv.lrs['lr-Adam/pg2'], "plot lr weights",
                save_path=os.path.join(trainer_retrain.log_dir, "lr_weights.pdf"))
    else:
        print("The LR was not correct and not plot was made")

    return

if __name__ == '__main__':
    # import pydevd_pycharm
    # pydevd_pycharm.settrace('localhost', port=12345, stdout_to_server=True, stderr_to_server=True, suspend=False)

    torch.cuda.empty_cache()
    from utils.util import prep_model_options
    model_options = prep_model_options()
    print(model_options)

    if model_options['model_name'] == 'ainet':
        seed_everything(2020, workers=True)
        print("the seed is 2020")
    elif model_options['model_name'] == 'scn' or model_options['model_name'] == 'cds' or model_options['model_name'] == 'ssm' :
        seed_everything(42, workers=True)
        print("the seed is 42")
    else:
        raise Exception
    torch.set_float32_matmul_precision('high')

    if model_options['pipeline']=='t_sep_v':
        assert model_options['weights_to_load'] is None, "We do not have weight loading for t_sep_v setup"
        assert model_options['weight_decay'] == 0, "For t sep v there should be NO weight decay"
        train_val_sep(hp_config=model_options)

    elif model_options['pipeline']=='tv':
        assert model_options['epochs'] > 0
        assert model_options['weight_decay'] > 0, "For TV there should be weight decay"
        train_val_combine(hp_config=model_options, max_epochs=model_options['epochs'])
    else:
        raise ValueError("Unknown pipeline {}".format(model_options['pipeline']))

import logging
from math import pi
from pathlib import Path
from typing import Callable, List, Optional

import bilby
import h5py
import numpy as np
import torch
from lightning.pytorch import Trainer, callbacks, loggers
from sampling import utils as sampling_utils

from ml4gw.distributions import Cosine, LogUniform, Uniform
from ml4gw.transforms import ChannelWiseScaler
from ml4gw.waveforms import SineGaussian
from mlpe.architectures import embeddings, flows
from mlpe.data.dataloader import PEInMemoryDataset
from mlpe.data.transforms import Preprocessor
from mlpe.data.transforms.injection import PEInjector
from mlpe.injection.priors import sg_uniform
from mlpe.logging import configure_logging
from typeo import scriptify

from .optimizers import optimizers
from .schedulers import schedulers
from .utils import split
from .validation import make_validation_dataset


class ParameterSampler(torch.nn.Module):
    def __init__(self, **parameters: Callable):
        super().__init__()
        self.parameters = parameters

    def forward(
        self,
        N: int,
        device: str = "cpu",
        
    ):
        parameters = {k: v(N).to(device) for k, v in self.parameters.items()}
        return parameters


def load_background(background_path: Path, ifos):
    background = []
    with h5py.File(background_path) as f:
        for ifo in ifos:
            hoft = f[ifo][:]
            background.append(hoft)
    return np.stack(background)


def load_signals(waveform_dataset: Path, parameter_names: List[str]):
    """
    Load in validation signals (generated with lalsimulation) and parameters
    """
    with h5py.File(waveform_dataset, "r") as f:
        signals = f["signals"][:]
        # TODO: how do we ensure order
        # of parameters throughout pipeline?
        parameters = []
        for param in parameter_names:
            if param == "phi":
                values = f["ra"][:]
                values = values - pi
            # take logarithm since hrss
            # spans large magnitude range
            elif param == "hrss":
                values = np.log10(f[param][:])
            else:
                values = f[param][:]

            parameters.append(values)

        parameters = np.column_stack(parameters)

    return signals, parameters


@scriptify(
    flow=flows,
    embedding=embeddings,
    optimizer=optimizers,
    scheduler=schedulers,
)
def main(
    background_path: Path,
    waveform_dataset: Path,
    flow: Callable,
    embedding: Callable,
    optimizer: Callable,
    scheduler: Callable,
    inference_params: List[str],
    ifos: List[str],
    sample_rate: float,
    kernel_length: float,
    fduration: float,
    highpass: float,
    batches_per_epoch: int,
    batch_size: int,
    device: str,
    outdir: Path,
    logdir: Path,
    testing_set: Path,
    max_epochs: int = 40,
    num_samples_draw: int = 3000,
    num_plot_corner: int = 20,
    early_stop: Optional[int] = None,
    valid_frac: Optional[float] = None,
    valid_stride: Optional[float] = None,
    verbose: bool = False,
    **kwargs,
):
    logdir.mkdir(exist_ok=True, parents=True)
    configure_logging(logdir / "train.log", verbose)

    n_params = len(inference_params)
    n_ifos = len(ifos)

    # load in background of shape (n_ifos, n_samples) and split into training
    # and validation if valid_frac specified
    background = load_background(background_path, ifos)

    logging.info(
        "Loading validation signals, performing train/val split of background "
        "and preparing waveform generator "
    )

    # load in the fixed set of validation waveforms
    # and split background into trainind and validation segments

    valid_signals, valid_parameters = load_signals(
        waveform_dataset, inference_params
    )

    if valid_frac is not None:
        background, valid_background = split(background, 1 - valid_frac, 1)

    # note: we pass the transpose the intrinsic parameters here because
    # the ml4gw transforms expects an array of shape (n_signals, n_params)

    # TODO: parameterize this somehow
    dec = Cosine()
    psi = Uniform(0, pi)
    phi = Uniform(-pi, pi)

    # intrinsic parameter sampler
    parameter_sampler = ParameterSampler(
        frequency=Uniform(32, 1024),
        quality=Uniform(2, 100),
        hrss=LogUniform(1e-23, 5e-20),
        phase=Uniform(0, 2 * pi),
        eccentricity=Uniform(0, 1),
    )

    # prepare waveform injector
    waveform = SineGaussian(sample_rate=sample_rate, duration=4)

    # prepare injector
    injector = PEInjector(
        sample_rate,
        ifos,
        parameter_sampler,
        dec,
        psi,
        phi,
        waveform,
    )

    parameter_sampler.to(device)
    waveform.to(device)
    injector.to(device)

    # sample parameters from parameter sampler
    # so we can fit the standard scaler
    samples = parameter_sampler(100000)
    samples["dec"] = dec(100000)
    samples["psi"] = psi(100000)
    samples["phi"] = phi(100000)

    parameters = []
    for param in inference_params:
        values = samples[param]
        if param == "hrss":
            values = np.log10(values)
        parameters.append(values)
    parameters = np.row_stack(parameters)

    standard_scaler = ChannelWiseScaler(n_params)
    preprocessor = Preprocessor(
        n_ifos,
        sample_rate,
        fduration,
        scaler=standard_scaler,
    )

    # create full training dataloader
    train_dataset = PEInMemoryDataset(
        background,
        int(kernel_length * sample_rate),
        batch_size=batch_size,
        batches_per_epoch=batches_per_epoch,
        preprocessor=injector,
        coincident=False,
        shuffle=True,
        device=device,
    )

    preprocessor.whitener.fit(
        kernel_length, highpass=highpass, sample_rate=sample_rate, *background
    )
    preprocessor.whitener.to(device)

    # to perform the normalization over each parameters,
    # the ml4gw ChannelWiseScaler expects an array of shape
    # (n_params, n_signals), so we pass the untransposed
    # intrinsic parameters here
    preprocessor.scaler.fit(parameters)
    preprocessor.scaler.to(device)

    # TODO: this light preprocessor wrapper can probably be removed
    # save preprocessor
    preprocess_dir = outdir / "preprocessor"
    preprocess_dir.mkdir(exist_ok=True, parents=True)
    torch.save(
        preprocessor.whitener.state_dict(), preprocess_dir / "whitener.pt"
    )
    torch.save(preprocessor.scaler.state_dict(), preprocess_dir / "scaler.pt")

    logging.info("Constructing validation dataloader")
    # construct validation dataset
    valid_dataset = None
    if valid_frac is not None:
        valid_dataset = make_validation_dataset(
            valid_background,
            valid_signals,
            valid_parameters,
            ifos,
            kernel_length,
            valid_stride,
            sample_rate,
            batch_size,
            device,
        )
    logging.info("Preparing Priors")
    priors = sg_uniform()
    priors["phi"] = bilby.core.prior.Uniform(
        name="phi", minimum=-np.pi, maximum=np.pi, latex_label="phi"
    )  # FIXME: remove when prior is moved to using torch tools
    logging.info("Loading test dataloaders")
    test_dataloader, _, _ = sampling_utils.initialize_data_loader(
        testing_set, inference_params, device
    )

    logging.info(f"Device: {device}")
    logging.info("set_float32_matmul_precision to high")
    torch.set_float32_matmul_precision("high")
    outdir.mkdir(exist_ok=True)

    strain, parameters = next(iter(train_dataset))
    param_dim = parameters.shape[-1]
    _, n_ifos, strain_dim = strain.shape
    logging.info("Building and initializing model")

    embedding = embedding((n_ifos, strain_dim))

    flow_obj = flow(
        (param_dim, n_ifos, strain_dim),
        embedding,
        preprocessor,
        optimizer,
        scheduler,
        inference_params,
        priors,
    )

    logging.info("Launching training")
    early_stop_cb = callbacks.EarlyStopping(
        "valid_loss", patience=early_stop, check_finite=True, verbose=True
    )
    lr_monitor = callbacks.LearningRateMonitor(logging_interval="epoch")
    logger = loggers.CSVLogger(save_dir=outdir / "pl-logdir", name="sg-model")
    trainer = Trainer(
        max_epochs=max_epochs,
        log_every_n_steps=50,
        accelerator=device,
        callbacks=[early_stop_cb, lr_monitor],
        logger=logger,
        gradient_clip_val=5.0,
    )
    trainer.fit(flow_obj, train_dataset, valid_dataset)
    logging.info(
        "Drawing {} samples for each test data".format(num_samples_draw)
    )
    trainer.test(flow_obj, test_dataloader, ckpt_path="last")

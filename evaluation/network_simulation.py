"""
Simulation of 4G LTE and 5G URLLC network latency and packet loss.
Provides functions to add latency to data packets and mask lost packets.
"""
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class NetworkConfig:
    """Configuration for network simulation."""
    name: str
    distribution: str  # 'gamma' or 'lognormal'
    params: dict       # e.g., {'shape': 2.0, 'scale': 4.0} for gamma
    deadline_ms: float = 50.0  # packets exceeding this are considered lost

# Predefined network profiles from paper
NETWORK_PROFILES = {
    '5G_URLLC': NetworkConfig(
        name='5G URLLC',
        distribution='gamma',
        params={'shape': 2.0, 'scale': 4.0},
        deadline_ms=50.0
    ),
    '4G_LTE': NetworkConfig(
        name='4G LTE',
        distribution='lognormal',
        params={'mean': np.log(40), 'sigma': 0.1},
        deadline_ms=50.0
    )
}

def simulate_network_latency(packet_timestamps: np.ndarray,
                             config: NetworkConfig,
                             random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate network latency for a sequence of packets.

    Args:
        packet_timestamps: array of packet transmission times (in ms, can be 0,1,2,...).
        config: NetworkConfig object.
        random_seed: seed for reproducibility.

    Returns:
        received_timestamps: array of arrival times (original + latency) for packets that arrive.
        mask: boolean array of same length as packet_timestamps, True if packet arrived within deadline.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_packets = len(packet_timestamps)
    if config.distribution == 'gamma':
        latency = np.random.gamma(shape=config.params['shape'],
                                  scale=config.params['scale'],
                                  size=n_packets)
    elif config.distribution == 'lognormal':
        # Note: params mean and sigma are for the underlying normal distribution
        latency = np.random.lognormal(mean=config.params['mean'],
                                      sigma=config.params['sigma'],
                                      size=n_packets)
    else:
        raise ValueError(f"Unknown distribution: {config.distribution}")

    # Determine which packets exceed deadline
    mask = latency <= config.deadline_ms
    received_timestamps = packet_timestamps + latency
    return received_timestamps, mask

def apply_network_to_batch(data: np.ndarray,
                           config: NetworkConfig,
                           sampling_rate: float = 1.0,  # Hz
                           random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate network transmission for a batch of patient episodes.
    For each time step in each episode, treat as a packet and apply latency.
    Packets that exceed deadline are set to NaN (to be handled by masking).

    Args:
        data: array of shape (n_patients, T, N) of vital signs at 1 Hz.
        config: NetworkConfig.
        sampling_rate: sampling rate (should be 1 Hz).
        random_seed:

    Returns:
        corrupted_data: data with NaN for lost packets.
        mask_per_packet: boolean array of same shape, True if packet arrived.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_patients, T, N = data.shape
    # Create packet timestamps: for each patient, time steps 0..T-1 (in seconds = ms*1000? Actually latency is in ms)
    # We'll treat each time step as a packet transmitted at that moment (in ms)
    packet_times = np.arange(T) * 1000  # convert seconds to ms

    corrupted = data.copy()
    mask_per_packet = np.ones_like(data, dtype=bool)

    for p in range(n_patients):
        for t in range(T):
            # Generate latency for this packet
            if config.distribution == 'gamma':
                latency = np.random.gamma(shape=config.params['shape'],
                                          scale=config.params['scale'])
            else:
                latency = np.random.lognormal(mean=config.params['mean'],
                                              sigma=config.params['sigma'])
            if latency > config.deadline_ms:
                corrupted[p, t, :] = np.nan
                mask_per_packet[p, t, :] = False

    return corrupted, mask_per_packet
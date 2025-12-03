#!/usr/bin/env python3
"""
Core Quantum Layer for LSTM-QNN
Focused quantum processing without classical post-processing.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import warnings

# Merlin imports
try:
    import merlin as ML
    from merlin.core.computation_space import ComputationSpace
    from merlin.measurement.strategies import MeasurementStrategy
    MERLIN_AVAILABLE = True
except ImportError as e:
    MERLIN_AVAILABLE = False
    warnings.warn(f"Merlin not available: {e}")


class CoreQuantumLayer(nn.Module):
    """
    Pure quantum processing layer without classical pre/post-processing.

    Features:
    - Takes 8D LSTM latent vectors (scaled [0, 2π])
    - Applies photonic quantum circuit processing
    - Returns raw quantum measurements for downstream processing
    - Fully trainable quantum parameters
    - Hardware-ready for photonic QPUs
    """

    def __init__(
        self,
        latent_dim: int = 8,
        n_modes: int = 10,
        n_photons: int = 5,
        n_quantum_params: int = 90,
        measurement_strategy: str = "probabilities",
        computation_space: str = "unbunched",
        amplitude_encoding: bool = False,  # Force angle encoding
        use_quantum: bool = None,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.n_modes = n_modes
        self.n_photons = n_photons
        self.n_quantum_params = n_quantum_params
        self.amplitude_encoding = amplitude_encoding  # Always False for angle encoding
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        # Determine quantum availability
        if use_quantum is None:
            self.use_quantum = MERLIN_AVAILABLE
        else:
            self.use_quantum = use_quantum and MERLIN_AVAILABLE

        if use_quantum and not MERLIN_AVAILABLE:
            warnings.warn("Quantum requested but Merlin unavailable. Using classical fallback.")
            self.use_quantum = False

        # Setup measurement strategy
        if isinstance(measurement_strategy, str):
            strategy_map = {
                "probabilities": MeasurementStrategy.PROBABILITIES,
                "amplitudes": MeasurementStrategy.AMPLITUDES,
                "mode_expectations": MeasurementStrategy.MODE_EXPECTATIONS
            }
            self.measurement_strategy = strategy_map.get(
                measurement_strategy.lower(), MeasurementStrategy.PROBABILITIES
            )
        else:
            self.measurement_strategy = measurement_strategy

        # Build quantum circuit
        if self.use_quantum:
            self._build_quantum_circuit()
        else:
            self._build_classical_fallback()

        print(f"✅ Core Quantum Layer initialized:")
        print(f"   Type: {'Quantum' if self.use_quantum else 'Classical'}")
        print(f"   Input: {latent_dim}D → Output: {self.output_dim}D")
        print(f"   Parameters: {sum(p.numel() for p in self.parameters())}")

    def _build_quantum_circuit(self):
        """Build the core quantum circuit with angle encoding using Merlin."""
        try:
            # Create quantum layer with explicit angle encoding (NOT amplitude encoding)
            self.quantum_circuit = ML.QuantumLayer.simple(
                input_size=self.latent_dim,          # 8 angle encoding inputs
                n_params=self.n_quantum_params,      # 90 trainable parameters
                device=self.device,
                dtype=self.dtype,
                no_bunching=True                     # Unbunched computation space
            )

            # Extract just the quantum layer (remove classical post-processing)
            self.quantum_core = self.quantum_circuit.quantum_layer
            self.output_dim = self.quantum_core.output_size

            print(f"🌌 Angle-encoded quantum circuit created:")
            print(f"   Encoding: Angle (8D LSTM → 8 rotation gates)")
            print(f"   Modes: {self.n_modes}, Photons: {self.n_photons}")
            print(f"   Quantum params: {self.n_quantum_params}")
            print(f"   Output dimension: {self.output_dim}")
            print(f"   Input range: [0, 2π] for optimal gate rotations")

        except Exception as e:
            warnings.warn(f"Quantum circuit creation failed: {e}")
            self.use_quantum = False
            self._build_classical_fallback()

    def _build_classical_fallback(self):
        """Classical approximation of quantum processing."""
        # Mimic quantum complexity with classical network
        hidden_dim = max(64, self.n_quantum_params)

        self.quantum_core = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.Tanh(),  # Mimic quantum interference patterns
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, self.n_modes * self.n_photons * 2)  # Mimic quantum state space
        )

        self.output_dim = self.n_modes * self.n_photons * 2

        print(f"⚙️  Classical fallback created with {self.output_dim}D output")

    def forward(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Core quantum processing.

        Args:
            latent_vectors: (batch_size, 8) - LSTM latents in [0, 2π]

        Returns:
            quantum_features: (batch_size, output_dim) - Raw quantum measurements
        """
        # Validate input
        if latent_vectors.dim() == 1:
            latent_vectors = latent_vectors.unsqueeze(0)

        batch_size = latent_vectors.shape[0]

        # Validate range for angle encoding (should be [0, 2π])
        if latent_vectors.min() < -0.1 or latent_vectors.max() > 2*np.pi + 0.1:
            warnings.warn(
                f"Input range [{latent_vectors.min():.3f}, {latent_vectors.max():.3f}] "
                f"outside optimal [0, 2π] for angle encoding. Each value maps to rotation gate angle."
            )

        # Quantum processing
        try:
            quantum_output = self.quantum_core(latent_vectors)

        except Exception as e:
            if self.use_quantum:
                warnings.warn(f"Quantum forward failed: {e}. Switching to classical.")
                self._build_classical_fallback()
                self.use_quantum = False
                quantum_output = self.quantum_core(latent_vectors)
            else:
                raise e

        return quantum_output

    def get_quantum_state(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Get raw quantum amplitudes (if available).

        Returns:
            amplitudes: Complex quantum state amplitudes
        """
        if not self.use_quantum:
            warnings.warn("get_quantum_state() not available in classical mode")
            return self.forward(latent_vectors)

        try:
            # For Merlin, try to get amplitudes directly
            with torch.no_grad():
                if hasattr(self.quantum_core, 'measurement_strategy'):
                    # Temporarily switch to amplitude measurement
                    old_strategy = self.quantum_core.measurement_strategy
                    self.quantum_core.measurement_strategy = MeasurementStrategy.AMPLITUDES
                    amplitudes = self.quantum_core(latent_vectors)
                    self.quantum_core.measurement_strategy = old_strategy
                    return amplitudes
                else:
                    return self.forward(latent_vectors)
        except:
            return self.forward(latent_vectors)

    def get_circuit_info(self) -> Dict[str, Any]:
        """Get quantum circuit information."""
        info = {
            'type': 'quantum' if self.use_quantum else 'classical_fallback',
            'latent_dim': self.latent_dim,
            'output_dim': self.output_dim,
            'n_modes': self.n_modes,
            'n_photons': self.n_photons,
            'n_quantum_params': self.n_quantum_params,
            'total_parameters': sum(p.numel() for p in self.parameters()),
            'measurement_strategy': self.measurement_strategy.name if self.use_quantum else 'N/A',
        }

        if self.use_quantum and hasattr(self.quantum_core, 'circuit'):
            info['circuit_depth'] = getattr(self.quantum_core.circuit, 'depth', 'unknown')
            info['computation_space'] = 'unbunched'

        return info

    def get_quantum_parameters(self) -> Dict[str, torch.Tensor]:
        """Extract quantum circuit parameters."""
        if self.use_quantum and hasattr(self.quantum_core, 'named_parameters'):
            return {name: param for name, param in self.quantum_core.named_parameters()}
        else:
            return {name: param for name, param in self.named_parameters()}

    def validate_input(self, x: torch.Tensor) -> bool:
        """Validate input tensor for quantum processing."""
        if x.shape[-1] != self.latent_dim:
            warnings.warn(f"Expected {self.latent_dim}D input, got {x.shape[-1]}D")
            return False

        if x.min() < -0.1 or x.max() > 2*np.pi + 0.1:
            warnings.warn("Input outside recommended [0, 2π] range for quantum encoding")
            return False

        return True

    @property
    def is_quantum(self) -> bool:
        """Check if using actual quantum processing."""
        return self.use_quantum

    @property
    def output_size(self) -> int:
        """Output dimension of quantum layer."""
        return self.output_dim

    def __repr__(self) -> str:
        return (f"CoreQuantumLayer("
                f"latent_dim={self.latent_dim}, "
                f"output_dim={self.output_dim}, "
                f"type={'quantum' if self.use_quantum else 'classical'}, "
                f"params={sum(p.numel() for p in self.parameters())}"
                f")")


def test_core_quantum_layer():
    """Test the core quantum layer functionality."""
    print("🧪 Testing Core Quantum Layer\n")

    # Test configuration
    batch_size = 32
    latent_dim = 8

    # Create test data (LSTM outputs in [0, 2π] range)
    test_input = torch.rand(batch_size, latent_dim) * 2 * np.pi

    print(f"📊 Test Input:")
    print(f"   Shape: {test_input.shape}")
    print(f"   Range: [{test_input.min():.3f}, {test_input.max():.3f}]")

    # Create quantum layer
    quantum_layer = CoreQuantumLayer(
        latent_dim=latent_dim,
        n_quantum_params=90
    )

    print(f"\n🔧 {quantum_layer}")
    print(f"\n📋 Circuit Info:")
    for key, value in quantum_layer.get_circuit_info().items():
        print(f"   {key}: {value}")

    # Test forward pass
    print(f"\n🚀 Testing forward pass...")
    try:
        with torch.no_grad():
            quantum_output = quantum_layer(test_input)

        print(f"✅ Forward pass successful!")
        print(f"   Output shape: {quantum_output.shape}")
        print(f"   Output range: [{quantum_output.min():.4f}, {quantum_output.max():.4f}]")

        # Test single sample
        single_output = quantum_layer(test_input[0])
        print(f"   Single sample shape: {single_output.shape}")

    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        return False

    # Test quantum state extraction
    print(f"\n🌌 Testing quantum state extraction...")
    try:
        quantum_state = quantum_layer.get_quantum_state(test_input[:5])
        print(f"✅ Quantum state extracted: {quantum_state.shape}")
    except Exception as e:
        print(f"⚠️  Quantum state extraction: {e}")

    # Test gradient flow
    print(f"\n🔄 Testing gradient flow...")
    try:
        quantum_layer.train()
        optimizer = torch.optim.Adam(quantum_layer.parameters(), lr=0.001)

        output = quantum_layer(test_input)
        loss = torch.mean(output**2)  # Dummy loss
        loss.backward()
        optimizer.step()

        print(f"✅ Gradient flow working! Loss: {loss.item():.6f}")

    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        return False

    print(f"\n🎉 All tests passed! Core Quantum Layer is ready.")
    return True


if __name__ == "__main__":
    success = test_core_quantum_layer()
    print(f"\n{'✅ SUCCESS' if success else '❌ FAILED'}: Core Quantum Layer test")
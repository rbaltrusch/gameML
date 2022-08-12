# -*- coding: utf-8 -*-
"""Reinforcement learning using Tensorflow"""

from __future__ import annotations

import itertools
import json
import logging
import random
import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable, Sequence, Tuple, TypeVar

import numpy
import tensorflow as tf  # type: ignore

from src.model import Moves, NpArray, lookup_position
from src.simulation import Simulation
from src.util import Position

T = TypeVar("T")
Optimizer = Any
LossFunction = Any


def create_default_optimizer() -> Optimizer:
    """Returns default optimizer (Adam)"""
    return tf.keras.optimizers.Adam(learning_rate=0.000_25, clipnorm=1.0)


@dataclass
class EpsilonUpdater:
    """Updates random exploration parameters using epsilon-greedy algorithm"""

    epsilon: float = 1.0
    epsilon_min: float = 0.1
    epsilon_max: float = 1.0
    epsilon_greedy_frames: int = 100_000  # exploration frames

    def __post_init__(self):
        # Reduction interval for chance of random actions
        self.epsilon_interval: float = self.epsilon_max - self.epsilon_min

    def update_epsilon(self):
        """Decay probability of taking random action"""
        self.epsilon -= self.epsilon_interval / self.epsilon_greedy_frames
        self.epsilon = max(self.epsilon, self.epsilon_min)


@dataclass
class HistorySample:
    """Contains a sample of the state history for a TfModel"""

    actions: Sequence[int]
    states: NpArray
    next_states: NpArray
    rewards: Sequence[float]
    dones: tf.Tensor


@dataclass
class TfModel:  # pylint: disable=too-many-instance-attributes
    """Tensorflow Model for reinforcement learning"""

    num_actions: int
    input_size: Tuple[int, int]
    gamma: float = 0.99  # discount factor
    sample_size: int = 5
    target_running_reward: int = 1_000
    max_steps_per_episode: int = 100_000
    max_memory_length: int = 100_000
    max_episode_length: int = 100
    update_after_actions: int = 1  # train the model
    update_target_network: int = 10_000  # update the target model
    epsilon_updater: EpsilonUpdater = EpsilonUpdater()
    optimizer: Optimizer = field(default_factory=create_default_optimizer)
    loss_function: LossFunction = field(default_factory=tf.keras.losses.Huber)

    def __post_init__(self):
        self.actions: Deque[int] = deque(maxlen=self.max_memory_length)
        self.states: Deque[NpArray] = deque(maxlen=self.max_memory_length)
        self.next_states: Deque[NpArray] = deque(maxlen=self.max_memory_length)
        self.rewards: Deque[float] = deque(maxlen=self.max_memory_length)
        self.done_history: Deque[bool] = deque(maxlen=self.max_memory_length)
        self.episode_rewards: Deque[float] = deque(maxlen=self.max_episode_length)
        self.model: tf.keras.Model = None  # type: ignore
        self.model_target: tf.keras.Model = None  # type: ignore

    def init_models(self):
        """Initialises models"""
        # Q-value prediction model
        self.model = self._create_model()
        # to calculate Q-value loss and predict future rewards
        self.model_target = self._create_model()

    def _create_model(self) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=self.input_size)

        # Screen frame convolutions
        conv = tf.keras.layers.Convolution2D
        args = {"padding": "same", "activation": "relu"}
        layer1 = conv(filters=8, kernel_size=8, **args)(inputs)
        layer2 = conv(filters=5, kernel_size=4, **args)(layer1)
        layer3 = conv(filters=5, kernel_size=3, **args)(layer2)
        layer4 = tf.keras.layers.Flatten()(layer3)
        layer5 = tf.keras.layers.Dense(512, activation="relu")(layer4)
        action = tf.keras.layers.Dense(self.num_actions, activation="linear")(layer5)
        return tf.keras.Model(inputs=inputs, outputs=action)

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as file:
            json.dump([x.tolist() for x in self.model.get_weights()], file, indent=4)

    def train_model(self, simulation: Simulation):
        """Trains neural network using reinforcement learning"""
        running_reward = 0

        for episode in itertools.count():  # infinite loop
            simulation.reset()
            state: NpArray = simulation.init()
            episode_reward = 0

            for frame, _ in enumerate(range(1, self.max_steps_per_episode)):
                action = self._take_action(frame, state)
                self.epsilon_updater.update_epsilon()
                step_data = simulation.step(action)

                episode_reward += step_data.reward
                self.actions.append(action.value)
                self.states.append(state)
                self.next_states.append(step_data.next_state)
                self.done_history.append(step_data.done)
                self.rewards.append(step_data.reward)
                state = step_data.next_state

                self._update_training_model(frame, self.update_after_actions)
                if frame % self.update_target_network == 0:
                    self._update_target_model(running_reward, episode, frame)  # type: ignore

                if step_data.done:
                    print(f"{self.epsilon_updater.epsilon=}")
                    break

            self.episode_rewards.append(episode_reward)
            running_reward = statistics.mean(self.episode_rewards)
            print(f"{running_reward=}")
            if running_reward > self.target_running_reward:
                print(f"Reached target reward at {episode=}.")
                break

    def compute(self, values: NpArray) -> Position:
        """Computes the optimal move to take given the state"""
        move = self._compute_action(values)
        return lookup_position(move)

    def _take_action(self, frame: int, state: NpArray) -> Moves:
        # epsilon-greedy exploration
        if self.epsilon_updater.epsilon > random.random():
            index: int = numpy.random.choice(self.num_actions)  # type: ignore
            return list(Moves)[index]
        print(f"non random {frame=}")
        return self._compute_action(state)

    def _compute_action(self, values: NpArray) -> Moves:
        state_tensor = tf.convert_to_tensor(values)  # type: ignore
        state_tensor = tf.expand_dims(state_tensor, 0)  # type: ignore
        action_probs = self.model(state_tensor, training=False)  # type: ignore
        index: int = tf.argmax(action_probs[0]).numpy()  # type: ignore # best action
        return list(Moves)[index]

    def _update_training_model(self, frame: int, update_after_actions: int):
        if not len(self.done_history) > self.sample_size:
            return

        if frame % update_after_actions != 0:
            return

        sample = self._sample_histories()
        updated_q_values = self._compute_updated_q_values(sample)
        loss, tape = self._compute_loss(sample, updated_q_values)
        self._backpropagate(loss, tape)

    def _sample_histories(self) -> HistorySample:
        indices: Iterable[int] = numpy.random.choice(  # type: ignore
            range(len(self.done_history)), size=self.sample_size
        )

        def sample(iterable: Iterable[T]) -> Tuple[T]:
            return tuple(iterable[i] for i in indices)  # type: ignore

        return HistorySample(
            actions=sample(self.actions),  # type: ignore
            states=numpy.array(sample(self.states)),
            next_states=numpy.array(sample(self.next_states)),
            dones=tf.convert_to_tensor(  # type: ignore
                [float(x) for x in sample(self.done_history)]
            ),
            rewards=sample(self.rewards),  # type: ignore
        )

    def _compute_updated_q_values(self, sample: HistorySample) -> NpArray:
        future_rewards = self.model_target.predict(
            sample.next_states
        )  # using target model for stability
        expected_future_reward = tf.reduce_max(future_rewards, axis=1)  # type: ignore
        updated_q_values: NpArray = sample.rewards + self.gamma * expected_future_reward

        # set last value to -1 if final frame
        updated_q_values = updated_q_values * (1 - sample.dones) - sample.dones  # type: ignore
        return updated_q_values

    def _compute_loss(
        self, sample: HistorySample, updated_q_values: NpArray
    ) -> Tuple[float, tf.GradientTape]:
        # create mask to compute loss only for updated Q-values
        masks = tf.one_hot(sample.actions, self.num_actions)  # type: ignore

        with tf.GradientTape() as tape:
            q_values = self.model(sample.states)  # train model on states
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)  # type: ignore
            loss = self.loss_function(updated_q_values, q_action)
        return loss, tape

    def _backpropagate(self, loss: float, tape: tf.GradientTape) -> None:
        grads = tape.gradient(loss, self.model.trainable_variables)  # type: ignore
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))  # type: ignore

    def _update_target_model(
        self, total_reward: numpy.float64, episode: int, frame: int
    ):
        self.model_target.set_weights(self.model.get_weights())
        logging.info(
            "Epside %d, frame %d: total reward %f", episode, frame, total_reward
        )

from typing import Optional

import torch
from torch import nn

from probnmn.models import (
    ProgramGenerator,
    ProgramPrior,
    QuestionReconstructor,
    NeuralModuleNetwork,
)


class Reinforce(nn.Module):
    def __init__(self, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce_baseline = 0.0
        self._baseline_decay = baseline_decay

    def forward(self, inputs, reward):
        centered_reward = reward.detach() - self._reinforce_baseline
        self._reinforce_baseline += self._baseline_decay * centered_reward.mean().item()
        return inputs * centered_reward


class _NegativeElboWithReinforce(nn.Module):
    def __init__(self, beta: float = 0.1, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce = Reinforce(baseline_decay)
        self._beta = beta

    def forward(
        self,
        inference_loss: torch.FloatTensor,
        reconstruction_loss: torch.FloatTensor,
        prior_loss: torch.FloatTensor,
        extra_reinforce_reward: Optional[torch.FloatTensor] = None,
    ):
        # KL-divergence (fully monte carlo form)
        negative_reinforce_reward = reconstruction_loss + self._beta * (
            prior_loss - inference_loss
        )

        if extra_reinforce_reward is not None:
            negative_reinforce_reward = negative_reinforce_reward - extra_reinforce_reward
        reinforce_reward = -negative_reinforce_reward

        path_derivative_loss = inference_loss.mean()
        score_function_loss = self._reinforce(inference_loss, reinforce_reward).mean()
        nelbo = reconstruction_loss + self._beta * path_derivative_loss + score_function_loss
        return nelbo


class _ElboWithReinforce(nn.Module):
    def __init__(self, beta: float = 0.1, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce_baseline = torch.tensor(0.0)

        self._beta = beta
        self._baseline_decay = baseline_decay

    def forward(self, inference_likelihood, reconstruction_likelihood, reinforce_reward):
        # Detach the reward term, we don't want gradients to flow to through that and get
        # counted twice, once already counted through path derivative.
        reinforce_reward = reinforce_reward.detach()

        # Subtract moving average baseline to reduce variance.
        # shape: (batch_size, )
        centered_reward = reinforce_reward - self._reinforce_baseline

        # Path derivative and score function estimator components of KL-divergence.
        # shape: (batch_size, )
        path_derivative_inference_likelihood = inference_likelihood
        reinforce_estimator_inference_likelihood = centered_reward * inference_likelihood

        # shape: (batch_size, )
        kl_divergence = (
            -self._beta * path_derivative_inference_likelihood
            - reinforce_estimator_inference_likelihood
        )

        # shape: (batch_size, )
        fully_monte_carlo_elbo = reconstruction_likelihood - kl_divergence

        # Update moving average baseline.
        self._reinforce_baseline += self._baseline_decay * centered_reward.mean(dim=-1)

        return {
            "reconstruction_likelihood": reconstruction_likelihood.mean(),
            "kl_divergence": kl_divergence.mean(),
            "elbo": fully_monte_carlo_elbo.mean(),
            "reinforce_reward": reinforce_reward.mean(),
        }


class QuestionCodingElbo(_ElboWithReinforce):
    def __init__(
        self,
        program_generator: ProgramGenerator,
        question_reconstructor: QuestionReconstructor,
        program_prior: ProgramPrior,
        beta: float = 0.1,
        baseline_decay: float = 0.99,
    ):
        super().__init__(beta, baseline_decay)
        self._program_generator = program_generator
        self._question_reconstructor = question_reconstructor
        self._program_prior = program_prior

    def forward(self, question_tokens: torch.LongTensor):
        # Sample programs using the inference model (program generator).
        # Sample z ~ q_\phi(z|x'), shape: (batch_size, max_program_length)

        # keys: {"predictions", "loss"}
        program_generator_output_dict = self._program_generator(question_tokens)
        sampled_programs = program_generator_output_dict["predictions"]

        # keys: {"predictions", "loss"}
        question_reconstructor_output_dict = self._question_reconstructor(
            sampled_programs, question_tokens
        )

        # Gather components required to calculate REINFORCE reward.
        # shape: (batch_size, )
        logprobs_reconstruction = -question_reconstructor_output_dict["loss"]
        logprobs_generation = -program_generator_output_dict["loss"]
        logprobs_prior = -self._program_prior(sampled_programs)["loss"]

        # REINFORCE reward (R): ( \log{ (p_\theta (x'|z) * p(z) ^ \beta) / (q_\phi (z|x') ) })
        # Note that reward is made of log-probabilities, not loss (negative log-probabilities)
        # shape: (batch_size, )
        reinforce_reward = logprobs_reconstruction + self._beta * (
            logprobs_prior - logprobs_generation
        )

        return super().forward(logprobs_generation, logprobs_reconstruction, reinforce_reward)


class JointTrainingNegativeElbo(_NegativeElboWithReinforce):
    def __init__(
        self,
        program_generator: ProgramGenerator,
        question_reconstructor: QuestionReconstructor,
        program_prior: ProgramPrior,
        nmn: NeuralModuleNetwork,
        beta: float = 0.1,
        gamma: float = 10,
        baseline_decay: float = 0.99,
        objective: str = "ours",
    ):

        super().__init__(beta, baseline_decay)
        self._program_generator = program_generator
        self._question_reconstructor = question_reconstructor
        self._program_prior = program_prior
        self._nmn = nmn

        self._gamma = gamma
        self._objective = objective

    def forward(
        self,
        question_tokens: torch.LongTensor,
        image_features: torch.FloatTensor,
        answer_tokens: torch.LongTensor,
    ):
        # Sample programs using the inference model (program generator).
        # Sample z ~ q_\phi(z|x'), shape: (batch_size, max_program_length)

        # keys: {"predictions", "loss"}
        program_generator_output_dict = self._program_generator(question_tokens)
        sampled_programs = program_generator_output_dict["predictions"]

        # keys: {"predictions", "loss"}
        question_reconstructor_output_dict = self._question_reconstructor(
            sampled_programs, question_tokens
        )
        nmn_output_dict = self._nmn(image_features, sampled_programs, answer_tokens)

        if self._objective == "baseline":
            # Baseline objective only takes answer logprobs as reward.
            reinforce_reward = -nmn_output_dict["loss"]

            negative_elbo_loss = self._reinforce(
                program_generator_output_dict["loss"], reinforce_reward
            ).mean()
        else:
            # Gather components required to calculate REINFORCE reward.
            negative_logprobs_reconstruction = question_reconstructor_output_dict["loss"]
            negative_logprobs_generation = program_generator_output_dict["loss"]
            negative_logprobs_prior = self._program_prior(sampled_programs)["loss"]
            negative_logprobs_answers = nmn_output_dict["loss"]

            negative_elbo_loss = super().forward(
                negative_logprobs_generation,
                negative_logprobs_reconstruction,
                negative_logprobs_prior,
                (-negative_logprobs_answers * self._gamma),
            )

        return {"nmn_loss": nmn_output_dict["loss"].mean(), "elbo": -negative_elbo_loss.mean()}

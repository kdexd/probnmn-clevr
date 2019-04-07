import torch
from torch import nn

from probnmn.models import (
    ProgramGenerator,
    ProgramPrior,
    QuestionReconstructor,
    NeuralModuleNetwork,
)


class Reinforce(nn.Module):
    r"""
    A PyTorch module which applies REINFORCE to inputs using a specified reward, and internally
    keeps track of a decaying moving average baseline.

    Parameters
    ----------
    baseline_decay: float, optional (default = 0.99)
        Factor by which the moving average baseline decays on every call.
    """
    def __init__(self, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce_baseline = 0.0
        self._baseline_decay = baseline_decay

    def forward(self, inputs, reward):
        centered_reward = reward.detach() - self._reinforce_baseline
        self._reinforce_baseline += self._baseline_decay * centered_reward.mean().item()
        return inputs * centered_reward


class _ElboWithReinforce(nn.Module):
    def __init__(self, beta: float = 0.1, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce_baseline = torch.tensor(0.0)

        self._beta = beta
        self._baseline_decay = baseline_decay

    def _forward(self, inference_likelihood, reconstruction_likelihood, reinforce_reward):
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
            reinforce_estimator_inference_likelihood
            - self._beta * path_derivative_inference_likelihood
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

        # REINFORCE reward (R): \log (p_\theta (x'|z)           (reconstruction)
        #                     + \beta * \log (p(z))             (prior)
        #                     - \beta * \log (q_\phi (z|x'))    (inference)
        # Note that reward is made of log-probabilities, not loss (negative log-probabilities)
        # shape: (batch_size, )
        reinforce_reward = logprobs_reconstruction + self._beta * (
            logprobs_prior - logprobs_generation
        )

        return super()._forward(logprobs_generation, logprobs_reconstruction, reinforce_reward)


class JointTrainingElbo(_ElboWithReinforce):
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

            elbo_output_dict = {
                "reinforce_reward": self._reinforce(
                    program_generator_output_dict["loss"], reinforce_reward
                ).mean()
            }
        else:
            # Gather components required to calculate REINFORCE reward.
            # shape: (batch_size, )
            logprobs_reconstruction = -question_reconstructor_output_dict["loss"]
            logprobs_generation = -program_generator_output_dict["loss"]
            logprobs_prior = -self._program_prior(sampled_programs)["loss"]
            logprobs_answering = -nmn_output_dict["loss"]

            # REINFORCE reward (R): \log (p_\theta (x'|z)                (reconstruction)
            #                     + \beta * \log (p(z))                  (prior)
            #                     - \beta * \log (q_\phi (z|x'))         (inference)
            #                     + \gamma * \log (q_\phi_z (a'|z, i'))  (answering)
            # Note that reward is made of log-probabilities, not loss (negative log-probabilities)
            # shape: (batch_size, )
            reinforce_reward = logprobs_reconstruction + self._beta * (
                logprobs_prior - logprobs_generation
            )

            # keys: {"reconstruction_likelihood", "kl_divergence", "elbo", "reinforce_reward"}
            elbo_output_dict = super()._forward(
                logprobs_generation, logprobs_reconstruction, reinforce_reward
            )

        return {**elbo_output_dict, "nmn_loss": nmn_output_dict["loss"].mean()}

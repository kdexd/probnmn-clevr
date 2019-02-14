import torch
from torch import nn

from probnmn.models import ProgramGenerator, ProgramPrior, QuestionReconstructor


class ElboWithReinforce(nn.Module):
    def __init__(self,
                 program_generator: ProgramGenerator,
                 question_reconstructor: QuestionReconstructor,
                 program_prior: ProgramPrior,
                 beta: float = 0.1,
                 baseline_decay: float = 0.99):
        super().__init__()
        self.program_generator = program_generator
        self.question_reconstructor = question_reconstructor
        self.program_prior = program_prior

        self._beta = beta

        self._moving_average_baseline = torch.tensor(0.0)
        self._baseline_decay = baseline_decay

    def forward(self, question_tokens: torch.LongTensor):
        # Sample programs using the inference model (program generator).
        # Sample z ~ q_\phi(z|x'), shape: (batch_size, max_program_length)

        # keys: {"predictions", "loss"}
        program_generator_output_dict = self.program_generator(question_tokens)
        sampled_programs = program_generator_output_dict["predictions"]

        # keys: {"predictions", "loss"}
        question_reconstructor_output_dict = self.question_reconstructor(
            sampled_programs, question_tokens
        )
        reconstruction_likelihood = - question_reconstructor_output_dict["loss"].mean(dim=-1)

        # Fully Monte Carlo form of KL (\log{q_\phi(z|x')} || \log{p(z)})
        # Note that reward is made of log-probabilities, not loss (negative log-probabilities)
        # shape: (batch_size, )
        logprobs_reconstruction = - question_reconstructor_output_dict["loss"]
        logprobs_generation = - program_generator_output_dict["loss"]
        logprobs_prior = - self.program_prior(sampled_programs)["loss"]

        # REINFORCE reward (R): ( \log{ (p_\theta (x'|z) * p(z) ^ \beta) / (q_\phi (z|x') ) })
        # shape: (batch_size, )
        reinforce_reward = logprobs_reconstruction + \
                           self._beta * (logprobs_prior - logprobs_generation)
        # Detach the reward term, we don't want gradients to flow to through that and get
        # counted twice, once already counted through path derivative loss.
        reinforce_reward = reinforce_reward.detach()

        # Subtract moving average baseline to reduce variance.
        # shape: (batch_size, )
        centered_reward = reinforce_reward - self._moving_average_baseline

        # Path derivative and score function estimator components of KL-divergence.
        path_derivative_generation = logprobs_generation.mean(dim=-1)
        score_function_estimator = (centered_reward * logprobs_generation).mean(dim=-1)
        kl_divergence = self._beta * path_derivative_generation + score_function_estimator

        fully_monte_carlo_elbo = reconstruction_likelihood - kl_divergence

        # Update moving average baseline.
        # B := B + (1 - \delta * (R - B))
        self._moving_average_baseline += self._baseline_decay * centered_reward.mean(dim=-1)

        return {
            "reconstruction_likelihood": reconstruction_likelihood,
            "kl_divergence": kl_divergence,
            "elbo": fully_monte_carlo_elbo,
            "reinforce_reward": reinforce_reward.mean(dim=-1)
        }

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
        # Detach the reward term, we don't want gradients to flow to through it.
        centered_reward = reward.detach() - self._reinforce_baseline

        # Update moving average baseline.
        self._reinforce_baseline += self._baseline_decay * centered_reward.mean().item()
        return inputs * centered_reward


class _ElboWithReinforce(nn.Module):
    r"""
    A PyTorch Module to compute the Fully Monte Carlo form of Evidence Lower Bound, given the
    inference likelihood, reconstruction likelihood and a REINFORCE reward. Accepting any scalar
    as REINFORCE reward allows flexibility in Evidence lower ound objective - like we have an
    extra answer log-likelihood term during Joint Training.

    This class is not used directly, instead its extended classes :class:`QuestionCodingElbo` and
    :class:`JointTrainingElbo` are used in corresponding phases.

    Parameters
    ----------
    beta: float, optional (default = 0.1)
        KL co-efficient. Refer ``BETA`` in :class:`~probnmn.config.Config`.
    baseline_decay: float, optional (default = 0.99)
        Decay co-efficient for moving average REINFORCE baseline. Refer ``DELTA`` in
        :class:`~probnmn.config.Config`.
    """

    def __init__(self, beta: float = 0.1, baseline_decay: float = 0.99):
        super().__init__()
        self._reinforce = Reinforce(baseline_decay=baseline_decay)
        self._beta = beta

    def _forward(
        self,
        inference_likelihood: torch.Tensor,
        reconstruction_likelihood: torch.Tensor,
        reinforce_reward: torch.Tensor,
    ):

        # Path derivative and score function estimator components of KL-divergence.
        # shape: (batch_size, )
        path_derivative_inference_likelihood = inference_likelihood
        reinforce_estimator_inference_likelihood = self._reinforce(
            inference_likelihood, reinforce_reward
        )

        # shape: (batch_size, )
        kl_divergence = (
            reinforce_estimator_inference_likelihood
            - self._beta * path_derivative_inference_likelihood  # noqa: W503
        )

        # shape: (batch_size, )
        fully_monte_carlo_elbo = reconstruction_likelihood - kl_divergence

        return {
            "reconstruction_likelihood": reconstruction_likelihood.mean(),
            "kl_divergence": kl_divergence.mean(),
            "elbo": fully_monte_carlo_elbo.mean(),
            "reinforce_reward": reinforce_reward.mean(),
        }


class QuestionCodingElbo(_ElboWithReinforce):
    r"""
    A PyTorch module to compute Evidence Lower Bound for observed questions without (GT) program
    supervision. This implementation takes the Fully Monte Carlo form, and uses :class:`Reinforce`
    estimator for parameters of the inference model
    (:class:`~probnmn.models.program_generator.ProgramGenerator`).

    Parameters
    ----------
    program_generator: ProgramGenerator
        A :class:`~probnmn.models.program_generator.ProgramGenerator`, serves as inference model
        of the posterior (programs).
    question_reconstructor: QuestinReconstructor
        A :class:`~probnmn.models.question_reconstructor.QuestionReconstructor`, serves as
        reconstruction model of observed data (questions).
    program_prior: ProgramPrior
        A :class:`~probnmn.models.program_prior.ProgramPrior`, serves as prior of the posterior
        distribution (programs).
    beta: float, optional (default = 0.1)
        KL co-efficient. Refer ``BETA`` in :class:`~probnmn.config.Config`.
    baseline_decay: float, optional (default = 0.99)
        Decay co-efficient for moving average REINFORCE baseline. Refer ``DELTA`` in
        :class:`~probnmn.config.Config`.
    """

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
        program_generator_output_dict = self._program_generator(
            question_tokens, decoding_strategy="sampling"
        )
        sampled_programs = program_generator_output_dict["predictions"]

        # keys: {"predictions", "loss"}
        question_reconstructor_output_dict = self._question_reconstructor(
            sampled_programs, question_tokens, decoding_strategy="sampling"
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
    r"""
    A PyTorch module to compute Evidence Lower Bound for observed questions without (GT) program
    supervision with the added answer log-likelihood term in the bound, from Joint Training
    objective. This implementation takes the Fully Monte Carlo form, and uses :class:`Reinforce`
    estimator for parameters of the inference model
    (:class:`~probnmn.models.program_generator.ProgramGenerator`).

    Parameters
    ----------
    program_generator: ProgramGenerator
        A :class:`~probnmn.models.program_generator.ProgramGenerator`, serves as inference model
        of the posterior (programs).
    question_reconstructor: QuestinReconstructor
        A :class:`~probnmn.models.question_reconstructor.QuestionReconstructor`, serves as
        reconstruction model of observed data (questions).
    program_prior: ProgramPrior
        A :class:`~probnmn.models.program_prior.ProgramPrior`, serves as prior of the posterior
        distribution (programs).
    nmn: NeuralModuleNetwork
        A :class:`~probnmn.models.nmn.NeuralModuleNetwork`, for answer log-likelihood term in the
        objective.
    beta: float, optional (default = 0.1)
        KL co-efficient. Refer ``BETA`` in :class:`~probnmn.config.Config`.
    gamma: float, optional (default = 10)
        Answer log-likelihood scaling co-efficient. Refer ``GAMMA`` in
        :class:`~probnmn.config.Config`.
    baseline_decay: float, optional (default = 0.99)
        Decay co-efficient for moving average REINFORCE baseline. Refer ``DELTA`` in
        :class:`~probnmn.config.Config`.
    objective: str, optional (default = "ours")
        Training objective, "baseline" - REINFORCE reward would only have answer log-likelihood.
        "ours" - REINFORCE reward would have the full Evidence Lower Bound added.
    """

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
        program_generator_output_dict = self._program_generator(
            question_tokens, decoding_strategy="sampling"
        )
        sampled_programs = program_generator_output_dict["predictions"]

        # keys: {"predictions", "loss"}
        question_reconstructor_output_dict = self._question_reconstructor(
            sampled_programs, question_tokens, decoding_strategy="sampling"
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
            reinforce_reward = (
                logprobs_reconstruction
                + self._beta * logprobs_prior  # noqa: W503
                - self._beta * logprobs_generation  # noqa: W503
                + self._gamma * logprobs_answering  # noqa: W503
            )

            # keys: {"reconstruction_likelihood", "kl_divergence", "elbo", "reinforce_reward"}
            elbo_output_dict = super()._forward(
                logprobs_generation, logprobs_reconstruction, reinforce_reward
            )

        return {**elbo_output_dict, "nmn_loss": nmn_output_dict["loss"].mean()}

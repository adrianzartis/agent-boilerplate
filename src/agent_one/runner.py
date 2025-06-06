from agents import Runner
import logging

from .agent import summary_agent, followup_agent
from schemas import SummaryOutput, FollowupOutput

logger = logging.getLogger(__name__)


class MultiAgentOutput(FollowupOutput):
    summary: str


async def run(text: str) -> MultiAgentOutput:
    logger.info("Running summary_agent with input: %s", text)
    original_summary_instructions = summary_agent.instructions
    summary_agent.instructions = original_summary_instructions.replace("{{text}}", text)
    run_summary = await Runner.run(summary_agent, input="")
    summary_agent.instructions = original_summary_instructions
    logger.info("Summary output: %s", run_summary.final_output.summary)

    summary_text = run_summary.final_output.summary

    logger.info("Running followup_agent with summary: %s", summary_text)
    original_followup_instructions = followup_agent.instructions
    followup_agent.instructions = original_followup_instructions.replace("{{summary}}", summary_text)
    run_followup = await Runner.run(followup_agent, input="")
    followup_agent.instructions = original_followup_instructions
    logger.info(
        "Followup output: question=%s kind=%s",
        run_followup.final_output.question,
        run_followup.final_output.kind,
    )

    return MultiAgentOutput(
        question=run_followup.final_output.question,
        kind=run_followup.final_output.kind,
        summary=summary_text,
    )

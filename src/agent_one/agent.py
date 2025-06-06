from agents import Agent, ModelSettings, OpenAIChatCompletionsModel

from client_factory import get_agent_open_ai_client, get_agent_model_name
from prompts import load_prompt
from schemas import SummaryOutput, FollowupOutput


# shared model configuration
model = OpenAIChatCompletionsModel(
    model=get_agent_model_name("general"),
    openai_client=get_agent_open_ai_client("general"),
)

model_settings = ModelSettings(temperature=0.2, max_tokens=500)

# first agent
summary_agent = Agent(
    name="SummaryAgent",
    model=model,
    model_settings=model_settings,
    output_type=SummaryOutput,
    instructions=load_prompt("agent_one.summary"),
)

# second agent
followup_agent = Agent(
    name="FollowupAgent",
    model=model,
    model_settings=model_settings,
    output_type=FollowupOutput,
    instructions=load_prompt("agent_one.followup"),
)

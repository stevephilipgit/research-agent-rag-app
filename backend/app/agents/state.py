"""Agent state definition shared by the LangGraph nodes."""
from typing import Annotated, TypedDict

import operator
from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]

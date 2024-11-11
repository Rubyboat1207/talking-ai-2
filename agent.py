from dataclasses import dataclass
import abc
from enum import Enum
from typing import Callable, Optional
import uuid
from threading import Thread, Lock

from speech_provider import SpeechProvider


class AgentContext:
    guid: str
    value: str

    def __init__(self, value: str):
        self.guid = str(uuid.uuid4())
        self.value = value


class EnvironmentalContext(AgentContext):
    pass


class HumanContext(AgentContext):
    pass


class ToolCallContext(AgentContext):
    parameters: dict[str, ...]
    response_id: str | None

    def __init__(self, value: str, parameters: dict[str, ...]):
        super().__init__(value)
        self.parameters = parameters
        self.response_id = None


class ToolCallResponseContext(AgentContext):
    call_id: str

    def __init__(self, value: str, call_id: str):
        super().__init__(value)
        self.call_id = call_id


class AgentResponseContext(AgentContext):
    pass


class FinishReason(Enum):
    STOP = 0,
    TOOL_CALL = 1


@dataclass
class AgentResponse:
    text_response: str
    tool_calls: Optional[list[ToolCallContext]]
    finish_reason: FinishReason


@dataclass
class Action:
    parameter_schema: object
    description: str
    name: str
    func: Callable[[dict[str, ...]], str]


class ActionManager:
    """Manages all action related activities instead of the agent"""
    lifetime_ephemeral_group_count: int
    actions: dict[str, Action]
    ephemeral_groups: dict[int, list[Action]]
    forced_actions_queue: list[str]

    action_mutex: Lock

    def __init__(self):
        self.lifetime_ephemeral_group_count = 0
        self.actions = {}
        self.ephemeral_groups = {}
        self.forced_actions_queue = []
        self.action_mutex = Lock()

    def preform_action(self, call: ToolCallContext) -> ToolCallResponseContext:
        """Called only by agents. Will execute a function of a given name."""
        self.action_mutex.acquire()
        if call.value in self.actions:
            action = self.actions[call.value]
        else:
            ephemeral = self._find_action_name_in_ephemeral_group(call.value)
            if ephemeral is None:
                res_ctx = ToolCallResponseContext("action not recognised, try a different function name", call.guid)
                call.response_id = res_ctx.guid
                self.action_mutex.release()
                return res_ctx

            ephemeral_group_id, action = ephemeral
            del self.ephemeral_groups[ephemeral_group_id]

        if action.name in self.forced_actions_queue:
            self.forced_actions_queue.remove(action.name)

        try:
            resp = action.func(call.parameters)
        except Exception as error:
            res_ctx = ToolCallResponseContext("An error occurred {}".format(error), call.guid)
            call.response_id = res_ctx.guid
            self.action_mutex.release()
            return res_ctx

        res_ctx = ToolCallResponseContext(resp, call.guid)
        call.response_id = res_ctx.guid
        self.action_mutex.release()
        return res_ctx

    def _find_action_name_in_ephemeral_group(self, name: str) -> (int, Action) or None:
        """finds an action inside the ephemeral groups, if found returns its ephemeral group id and the associated action. Otherwise, returns None"""
        for idx, group in self.ephemeral_groups:
            for action in group:
                if action.name == name:
                    return idx, action

    def response_meets_action_criteria(self, response: AgentResponse):
        """Called only by agents. Will return whether a response uses all forced actions or any other future added requirements that may mean that the response is regenerated."""
        if len(self.forced_actions_queue) == 0:
            return True

        names = [call.value for call in response.tool_calls]

        for action in self.forced_actions_queue:
            if action not in names:
                return False

        return True

    def register_action(self, action: Action):
        """Adds the action to the internal registered actions dictionary."""
        self.actions[action.name] = action

    def unregister_action(self, name: str):
        """Removes the action of the given name from the internal registered actions dictionary."""
        del self.actions[name]

    def enqueue_forced_action(self, name: str):
        """Will force the action of the given name to be run before the next response."""
        if name not in self.forced_actions_queue:
            self.forced_actions_queue.append(name)

    def create_ephemeral_action_group(self, actions: list[Action]) -> int:
        """Returns an ephemeral group ID. An ephemeral group is removed once one is used. Good for making a decision."""
        self.ephemeral_groups[self.lifetime_ephemeral_group_count] = actions
        self.lifetime_ephemeral_group_count += 1

        return self.lifetime_ephemeral_group_count


class Agent(metaclass=abc.ABCMeta):
    def __init__(self, speech_provider: SpeechProvider):
        self._speech_provider = speech_provider
        self.action_manager = ActionManager()
        self._ctx: list[AgentContext] = []

    def add_context(self, agent_context: AgentContext):
        self._ctx.append(agent_context)

    @abc.abstractmethod
    def generate_response(self) -> (AgentResponse, str):
        pass

    def add_response_to_context(self, response: AgentResponse, execute_calls: bool = False,
                                execute_calls_async: bool = False):
        self.add_context(AgentResponseContext(response.text_response))

        if response.tool_calls is None:
            return

        for tool_call in response.tool_calls:
            self.add_context(tool_call)

        if not execute_calls:
            return

        for tool_call in response.tool_calls:
            def i_hate_internal_functions():
                self.add_context(self.action_manager.preform_action(tool_call))

            if execute_calls_async:
                thread = Thread(target=i_hate_internal_functions)

                thread.start()
            else:
                i_hate_internal_functions()

    def speak_recent_response(self):
        for entry in reversed(self._ctx):
            if isinstance(entry, AgentResponseContext):
                if entry.value == '':
                    return
                self._speech_provider.generate_speech(entry.value)
                return

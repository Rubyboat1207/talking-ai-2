import asyncio
import time

import websockets
import json
from uuid import uuid4 as uuid

from websockets import ConnectionClosedError
from websockets.asyncio.server import ServerConnection

from agent import Agent, Action, EnvironmentalContext, SystemPromptContext, HumanContext

class NeuroStyleActionCommand:
    ident: str
    name: str
    data: str or None

    def __init__(self, ident: str, name: str, data: str or None):
        self.ident = ident
        self.name = name
        self.data = data

    def to_json(self):
        if self.data is not None:
            return {"id": self.ident, "name": self.name, "data": self.data}
        else:
            return {"id": self.ident, "name": self.name}

class NeuroStyleResponse:
    command: str
    data: NeuroStyleActionCommand

    def __init__(self, data: NeuroStyleActionCommand, command: str="action"):
        self.command = command
        self.data = data

    def to_json(self):
        return {'command': self.command, 'data': self.data.to_json()}



def create_action(websocket_manager, action_name: str, websocket):
    async def execute(params: dict[str, ...]):
        cur_id = str(uuid())

        await websocket.send(json.dumps(NeuroStyleResponse(NeuroStyleActionCommand(cur_id, action_name, json.dumps(params))).to_json()))
        print(id(asyncio.get_running_loop()))

        future = asyncio.get_running_loop().create_future()
        websocket_manager.pending_actions[cur_id] = future

        try:
            # Await the future with a 10-second timeout
            response = await asyncio.wait_for(future, timeout=30.0)
        except asyncio.TimeoutError:
            # Handle timeout here if needed
            print(f"Action {action_name} {cur_id} timed out. ")
            response = "sorry, the action timed out."  # Or any default response you'd like to return on timeout
        finally:
            # Clean up pending actions regardless of timeout
            del websocket_manager.pending_actions[cur_id]

        return response

    return execute


class NeuroSamaWebsocketManager:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.connections = set()
        self.pending_actions: dict[str, asyncio.Future] = {}
        self.requests_action = asyncio.Event()

    async def manage(self, websocket: ServerConnection):
        self.connections.add(websocket)
        await asyncio.sleep(2)
        str_data = ""
        while True:
            try:
                str_data = await websocket.recv()
                print('received: ', str_data)
                inbound_data = json.loads(str_data)

                command = inbound_data['command']
                game = inbound_data['game']

                data: dict[str, ...] = inbound_data['data']

                if command == 'startup':
                    self.agent.add_context(EnvironmentalContext(f"You are now playing {game}."))
                elif command == 'context':
                    message = data['message']
                    silent = data['silent']
                    if silent:
                        self.agent.add_context(EnvironmentalContext(message))
                    else:
                        self.agent.add_context(HumanContext(message))
                elif command == 'actions/register':
                    for action in data['actions']:
                        self.agent.action_manager.register_action(self.generate_action_using_data(action, websocket))
                elif command == 'actions/unregister':
                    for action in data['action_names']:
                        self.agent.action_manager.unregister_action(action)
                elif command == 'actions/force':
                    self.agent.action_manager.action_force = data['action_names']

                    ctx = SystemPromptContext(data['query'] + " " + ("" if data.get('state') is None else data['state']))

                    if data.get('ephemeral_context') is not None:
                        ctx.ephemeral = data['ephemeral_context']

                    self.agent.add_context(ctx)
                elif command == 'action/result':
                    msg = data.get('message')
                    self.pending_actions[data['id']].set_result("" if msg is None else msg)
            except json.JSONDecodeError:
                print(f'panic! invalid JSON!!! got: {str_data}')
                exit(1)
            except ConnectionClosedError:
                break

    def generate_action_using_data(self, data, websocket):
        return Action(
            data['name'],
            data['description'],
            data['schema'],
            create_action(self, data['name'], websocket)
        )




    async def init_websocket(self):
        async def man(websocket):
            await self.manage(websocket)

        async with websockets.serve(man, "127.0.0.1", 9302):
            await asyncio.Future()

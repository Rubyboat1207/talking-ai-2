import asyncio
import time

import websockets
import json
from uuid import uuid4 as uuid

from websockets import ConnectionClosedError
from websockets.asyncio.server import ServerConnection

from agent import Agent, Action, EnvironmentalContext


def create_action(websocket_manager: "WebsocketManager", action_name: str, websocket):
    async def execute(params: dict[str, ...]):
        cur_id = str(uuid())

        await websocket.send(json.dumps(
            {
                'type': 'execute_action',
                'action_name': action_name,
                'action_id': cur_id,
                'params': json.dumps(params),
            }
        ))
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


class WebsocketManager:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.connections = set()
        self.pending_actions: dict[str, asyncio.Future] = {}
        self.requests_action = asyncio.Event()

    async def manage(self, websocket: ServerConnection):
        self.connections.add(websocket)
        print('send_all_actions requested')
        await asyncio.sleep(2)
        await websocket.send(json.dumps({'type': 'send_all_actions'}))

        while True:
            try:
                str_data = await websocket.recv()
                print('received: ', str_data)
                data = json.loads(str_data)

                path = data.get('path')

                if path == 'actions/register':
                    self.agent.action_manager.register_action(self.generate_action_using_data(data, websocket))
                elif path == 'actions/register/ephemeral':
                    actions = []
                    for action in data['actions']:
                        actions.append(self.generate_action_using_data(action, websocket))
                    self.agent.action_manager.create_ephemeral_action_group(actions)
                elif path == 'action/result':
                    action = self.pending_actions.get(data['action_id'])

                    if action is None:
                        await websocket.send(json.dumps({'ok': False, 'message': 'action_id does not exist'}))
                        continue

                    action.set_result(data['result'])
                    action.done()
                elif path == 'context/environment':
                    self.agent.add_context(EnvironmentalContext(data['value']))
                elif path == 'actions/request':
                    self.requests_action.set()
                elif path == 'actions/force':
                    self.agent.action_manager.enqueue_forced_action(data['name'])
                else:
                    await websocket.send(json.dumps({'ok': False, 'message': 'unknown message type'}))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"ok": False, "message": "Invalid JSON"}))
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

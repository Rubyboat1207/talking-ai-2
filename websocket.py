import asyncio
import websockets
import json
from uuid import uuid4 as uuid

from websockets import ConnectionClosedError
from websockets.asyncio.server import ServerConnection

from agent import Agent, Action


class WebsocketManager:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.connections = set()
        self.pending_actions: dict[str, asyncio.Future] = {}

    async def manage(self, websocket: ServerConnection):
        self.connections.add(websocket)
        await websocket.send(json.dumps({'type': 'send_all_actions'}))

        while True:
            try:

                data = json.loads(await websocket.recv())
                path = data.get('path')

                if path == 'actions/register':
                    async def execute(params: dict[str, ...]):
                        cur_id = str(uuid())

                        await websocket.send(json.dumps(
                            {
                                'type': 'execute_action',
                                'action_id': cur_id,
                                'params': json.dumps(params),
                            }
                        ))
                        print(id(asyncio.get_running_loop()))

                        future = asyncio.get_running_loop().create_future()

                        self.pending_actions[cur_id] = future

                        response = await future

                        del self.pending_actions[cur_id]

                        return response

                    action = Action(
                        data['name'],
                        data['description'],
                        data['schema'],
                        execute
                    )

                    self.agent.action_manager.register_action(action)

                    await websocket.send(json.dumps({'ok': True}))
                elif path == 'action/result':
                    action = self.pending_actions.get(data['action_id'])

                    if action is None:
                        await websocket.send(json.dumps({'ok': False, 'message': 'action_id does not exist'}))
                        continue

                    action.set_result(data['result'])
                    action.done()
                    await websocket.send(json.dumps({'ok': True}))
                else:
                    await websocket.send(json.dumps({'ok': False, 'message': 'unknown message type'}))
            except json.JSONDecodeError:
                await websocket.send(json.dumps({"ok": False, "message": "Invalid JSON"}))
            except ConnectionClosedError:
                break

    async def init_websocket(self):
        async def man(websocket):
            await self.manage(websocket)

        async with websockets.serve(man, "127.0.0.1", 9302):
            await asyncio.Future()

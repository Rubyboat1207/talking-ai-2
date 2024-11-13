import asyncio
import time
from threading import Thread
from dotenv import load_dotenv
from agent import HumanContext, Action, FinishReason, AgentContext, EnvironmentalContext, SystemPromptContext
from agents.openai_agent import OpenAiAgent
from speech_providers.console_output_speech_provider import ConsoleOutputSpeechProvider as Sp
from stt import stt
from websocket import WebsocketManager

load_dotenv()

agent = OpenAiAgent(Sp())

wait_run_without_human: asyncio.Future | None = None
on_env_ctx_added = -1

def on_context_added(ctx: AgentContext):
    global on_env_ctx_added
    if isinstance(ctx, EnvironmentalContext):
        on_env_ctx_added = time.time()

# async def monitoring_env_ctx():
#     global on_env_ctx_added
#     while True:
#         if on_env_ctx_added != -1 and time.time() - on_env_ctx_added > 4:
#             if wait_run_without_human is not None:
#                 wait_run_without_human.set_result(None)
#         await asyncio.sleep(1)

agent.context_added_notifiers.append(on_context_added)

agent.add_context(SystemPromptContext("You are a TTS ai. Keep responses speakable and short. Dont make lists. Be decisive. No markdown is allowed."))

async def main():
    global wait_run_without_human
    global on_env_ctx_added
    use_stt = False
    websocket_manager = WebsocketManager(agent)

    # Create the websocket task and await it in the background
    websocket_task = asyncio.create_task(websocket_manager.init_websocket())
    # asyncio.create_task(monitoring_env_ctx())

    loop = asyncio.get_running_loop()

    wait_run_without_human = loop.create_future()

    try:
        while True:
            # Run blocking input in an executor to avoid blocking the event loop
            # tasks = [asyncio.create_task(stt()), wait_run_without_human]
            # done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            # user_input = done.pop().result()
            # if user_input is None:
            #     print('sorry user, you have been interrupted.')
            #     pending.pop().cancel()
            #     on_env_ctx_added = -1
            #     wait_run_without_human = loop.create_future()
            # else:
            #     agent.add_context(HumanContext(user_input))
            if use_stt:
                text = await stt()
            else:
                text = await loop.run_in_executor(None, input, 'speak to it: ')
            print('you said ' + text)
            agent.add_context(HumanContext(text))

            res = None
            while res is None or res.finish_reason != FinishReason.STOP:
                res = agent.generate_response()
                await agent.add_response_to_context(res, True)

            agent.speak_recent_response()

    except asyncio.CancelledError:
        print("Main function cancelled, shutting down...")

    finally:
        # Ensure websocket_task is awaited and handled
        websocket_task.cancel()
        try:
            await websocket_task
        except asyncio.CancelledError:
            print("Websocket task cancelled and shut down gracefully.")


asyncio.run(main())

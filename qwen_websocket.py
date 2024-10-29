from threading import Thread

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, TextIteratorStreamer,AutoModelForCausalLM

import uvicorn

pretrained = "./Qwen1.5-1.8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(pretrained, trust_remote_code=True).half().cuda()
model = model.eval()
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
app = FastAPI()

app.add_middleware(
    CORSMiddleware
)

with open('websocket_demo.html',encoding="utf-8") as f:
    html = f.read()


@app.get("/")
async def get():
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    input: JSON String of {"query": "", "history": []}
    output: JSON String of {"response": "", "history": [], "status": 200}
        status 200 stand for response ended, else not
    """
    await websocket.accept()
    try:
        while True:
            json_request = await websocket.receive_json()
            query = json_request['query']
            history = json_request['history']
            for response, history in stream_chat(query, history=history):
                await websocket.send_json({
                    "response": response,
                    "history": history,
                    "status": 202,
                })
            await websocket.send_json({"status": 200})
    except WebSocketDisconnect:
        pass


def stream_chat(query, history):
    query={"role":"user","content":query}
    history.append(query)
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=512,
    #     streamer=streamer
    # )
    # generated_ids = [
    #     output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    # ]
    generation_kwargs = dict(model_inputs,streamer=streamer, max_new_tokens=512)
    thread=Thread(target=model.generate,kwargs=generation_kwargs)
    thread.start()
    output_text=""
    for new_text in streamer:
        output_text+=new_text
        yield new_text,history
    # response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    response=output_text
    history.append({"role": "assistant", "content": response})
    yield response,history


def main():
    uvicorn.run(f"{__name__}:app", host='0.0.0.0', port=8000, workers=1)


if __name__ == '__main__':
    main()

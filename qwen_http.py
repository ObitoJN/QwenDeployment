import time
import torch
import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Literal, Optional, Union
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from sse_starlette.sse import ServerSentEvent, EventSourceResponse

from vo import ModelList, ModelCard, ChatCompletionResponse, ChatCompletionRequest, ChatCompletionResponseChoice, \
    ChatMessage, DeltaMessage, ChatCompletionResponseStreamChoice

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"],
                   allow_headers=["*"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


@app.get("/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(id="Qwen1.5-8B")
    return ModelList(data=[model_card])


def chat(tokenizer, history):
    global model, streamer
    text = tokenizer.apply_chat_template(
        history,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=512,
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


@app.post("/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global model, tokenizer

    if request.messages[-1].role != "user":
        raise HTTPException(status_code=400, detail="Invalid request")
    # query = request.messages[-1].content

    # prev_message = request.messages[:-1]
    # if len(prev_message) > 0 and prev_message[0].role == "system":
    #     query = prev_message.pop(0).content + query
    #
    # history = []
    # if len(prev_message) % 2 == 0:
    #     for i in range(0, len(prev_message), 2):
    #         if prev_message[i].role == "user" and prev_message[i + 1].role == "assistant":
    #             history.append([prev_message[i].content, prev_message[i + 1].content])
    history = [{"role": message.role, "content": message.content} for message in request.messages]
    response = chat(tokenizer, history=history)
    choice_data = ChatCompletionResponseChoice(index=0, message=ChatMessage(role="assistant", content=response),
                                               finish_reason="stop")

    return ChatCompletionResponse(id="test",choices=[choice_data],object="chat.completion")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./Qwen1.5-1.8B-Chat", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("./Qwen1.5-1.8B-Chat", trust_remote_code=True).cuda()
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    # 多显卡支持，使用下面两行代替上面一行，将num_gpus改为你实际的显卡数量
    # from utils import load_model_on_gpus
    # model = load_model_on_gpus("THUDM/chatglm2-6b", num_gpus=2)
    model.eval()

    uvicorn.run(app, host='0.0.0.0', port=8000, workers=1)

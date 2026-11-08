from huggingface_hub import hf_hub_download
from trl.chat_template_utils import get_training_chat_template

def hermes(file):
    path = hf_hub_download("NousResearch/hermes-function-calling-v1", file, repo_type="dataset")
    return json.load(open(path, encoding="utf-8"))

single, glaive = hermes("func-calling-singleturn.json"), hermes("glaive-function-calling-5k.json")
print(single[0]["conversations"][2]["value"][:150])       # a Hermes reply: the call is JSON text

def hermes_to_chat(row):
    # The Hermes text writes calls as JSON inside <tool_call> tags. Turn them into structured tool_calls,
    # so the model's own chat template renders them in the model's own format.
    tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
    messages = []
    for turn in row["conversations"]:
        role, text = turn["from"], turn["value"]
        if role == "system":
            continue                                   # the template writes its own tool prompt from the tools list
        if role == "human":
            messages.append({"role": "user", "content": text})
        elif role == "tool":
            body = text.replace("<tool_response>", "").replace("</tool_response>", "").strip()
            messages.append({"role": "tool", "content": body})
        elif role == "gpt":
            calls = []
            for block in re.findall("<tool_call>(.*?)</tool_call>", text, re.S):
                call = json.loads(block)               # raises on a malformed call; the caller drops the row
                calls.append({"type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}})
            content = re.sub("<tool_call>.*?</tool_call>", "", text, flags=re.S).strip()
            msg = {"role": "assistant", "content": content}
            if calls:
                msg["tool_calls"] = calls
            messages.append(msg)
    return {"messages": messages, "tools": tools or []}

print(hermes_to_chat(single[0])["messages"][1]["tool_calls"][0])

train_tpl = get_training_chat_template(tok)        # Qwen3.5's template with generation markers, shipped in TRL
def render(chat):
    return list(tok.apply_chat_template(chat["messages"], tools=chat["tools"], chat_template=train_tpl)["input_ids"])

N_PER_SOURCE = 600
train = []
for name, rows in [("singleturn", single), ("glaive", glaive)]:
    kept, dropped = C.select_rows(rows, N_PER_SOURCE, render, max_tokens=1536)
    train += kept
    print(name, "| kept", len(kept), "| dropped", dropped)
bad = sum(1 for row in single if not C.parses(row))
print(len(train), "training conversations | singleturn rows with a call that is not valid JSON:", bad, "of", len(single))

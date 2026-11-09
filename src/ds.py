import json,sys,urllib.request
K=open("/tmp/claude-0/-home-user/fe3dc421-8c70-5172-bf54-6224b620a519/scratchpad/ds_key").read().strip()
def ask(prompt,system="You are a rigorous ML research assistant. Be concise.",model="deepseek-chat",max_tokens=3000):
    r=urllib.request.Request("https://api.deepseek.com/chat/completions",data=json.dumps({"model":model,"messages":[{"role":"system","content":system},{"role":"user","content":prompt}],"max_tokens":max_tokens,"temperature":0.3}).encode(),headers={"Authorization":"Bearer "+K,"Content-Type":"application/json"})
    return json.load(urllib.request.urlopen(r,timeout=300))["choices"][0]["message"]["content"]
if __name__=="__main__": print(ask(open(sys.argv[1]).read() if len(sys.argv)>1 else "Reply OK."))

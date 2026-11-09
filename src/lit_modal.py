import modal
app=modal.App("plantid-lit")
img=modal.Image.debian_slim(python_version="3.11").add_local_file("/home/user/plantid/src/lit.py","/root/lit.py")
@app.function(image=img,timeout=1800)
def run():
    import subprocess; r=subprocess.run(["python","/root/lit.py","/tmp/p.json"],capture_output=True,text=True); return r.stdout[-500:]+r.stderr[-500:],open("/tmp/p.json").read()
@app.local_entrypoint()
def main():
    log,data=run.remote(); open("/home/user/plantid/ideation/papers.json","w").write(data); print("LIT",log)

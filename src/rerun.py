from extract import app, run
from round1 import r1
@app.local_entrypoint()
def rerun_main():
    import json
    print(list(run.map(["dinov3_b","dinov3_l"])),flush=True)
    res=list(r1.map(["B","L"])); json.dump(res,open("../results/r1.json","w"),default=str); print("R1 DONE",flush=True)

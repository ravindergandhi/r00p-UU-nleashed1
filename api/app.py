from flask import Flask, request
from gradio_client import Client, file

app = Flask(__name__)


@app.route("/")
def welcome():
    return "roop-unleashed WebAPI Started!"


@app.route("/handle")
def handle():

    srcFiles = request.args.get("src")
    destFiles = request.args.get("dest")

    client = Client("http://127.0.0.1:7860/")

    client.predict(
        srcfiles=[file("http://127.0.0.1:7860/file=" + srcFiles)],
        api_name="/on_srcfile_changed",
    )
    client.predict(
        destfiles=[file("http://127.0.0.1:7860/file=" + destFiles)],
        api_name="/on_destfiles_changed",
    )
    result = client.predict(
        enhancer="None",  # Codeformer
        detection="First found",
        keep_frames=False,
        wait_after_extraction=False,
        skip_audio=False,
        face_distance=0.65,
        blend_ratio=0.65,
        selected_mask_engine="DFL XSeg",
        clip_text="cup,hands,hair,banana",
        processing_method="In-Memory processing",
        no_face_action="Use untouched original frame",
        vr_mode=False,
        autorotate=True,
        num_swap_steps=1,
        imagemask=None,
        api_name="/start_swap",
    )
    print(result)

    return "success"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=10050)

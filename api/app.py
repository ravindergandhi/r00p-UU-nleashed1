from flask import Flask, request
from gradio_client import Client, file

app = Flask(__name__)


@app.route("/")
def welcome():
    return "roop-unleashed WebAPI Started!"


@app.route("/handle")
def handle():

    srcFiles = request.args.getlist("src")
    destFiles = request.args.get("dest")

    client = Client("http://127.0.0.1:7860/")

    # 设置人脸照
    for srcFile in srcFiles:
        client.predict(
            srcfiles=[file("http://127.0.0.1:7860/file=" + srcFile)],
            api_name="/on_srcfile_changed",
        )

    # 设置目标照片
    client.predict(
        destfiles=[file("http://127.0.0.1:7860/file=" + destFiles)],
        api_name="/on_destfiles_changed",
    )

    # 设置目标人脸
    face_from_selected_res = client.predict(
        files=[file("http://127.0.0.1:7860/file=" + destFiles)],
        frame_num=1,
        api_name="/on_use_face_from_selected",
    )
    print("===== face_from_selected_res =====")
    print(face_from_selected_res)
    list_element = face_from_selected_res[0]

    for index, dictionary in enumerate(list_element):
        image_path = dictionary.get("image")
        print("Image path:", image_path)
        # 设置人脸选中
        client.predict(index=index, api_name="/set_selected_face_index")
        client.predict(api_name="/on_selected_face")

    # 开始处理
    result = client.predict(
        enhancer="Codeformer",  # Codeformer None
        detection="Selected face",
        keep_frames=False,
        wait_after_extraction=False,
        skip_audio=False,
        face_distance=0.01,
        blend_ratio=0.85,
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
    print("===== start_swap =====")
    print(result)

    # 清空人脸照列表
    client.predict(
        api_name="/on_clear_input_faces",
    )
    # 清空目标照列表
    client.predict(
        api_name="/clean_list_files_process",
    )
    client.predict(
        api_name="/on_clear_destfiles",
    )

    return "success"


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=10050)

import io

from fastapi import FastAPI, File
from starlette.responses import StreamingResponse

from rest_image_tracker.detector import Detector


app = FastAPI(
    title='Rest image tracker',
    description='Application that allows to post an image and find people in that image via REST API.',
    version='0.0.1',
)


@app.get('/')
def hello_world_endpoint() -> dict[str, str]:
    """
    Return simple hello world response

    :return: hello message
    """
    return {'msg': 'Hello world'}


@app.post('/detect')
def detector_endpoint(file: bytes = File()) -> StreamingResponse:
    """
    Return response with processed image

    :param file: image to process
    :return: processed image
    """
    detector = Detector()
    detector.load_img_from_bytes(file)
    detector.perform_detecting('HOG')
    return StreamingResponse(
        io.BytesIO(detector.encode_image()),
        media_type="image/jpg",
    )

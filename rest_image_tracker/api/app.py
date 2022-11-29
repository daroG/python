from fastapi import FastAPI


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

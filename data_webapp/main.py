import argparse
import os

import aiohttp_cors
import torch
from aiohttp import web

from translation.knn import KNNModel
from translation.manager import Manager
from translation.translate import translate

PYTORCH_MODEL = 'data/en-de.pt'
KNN_MODEL = 'data/neighbors.pickle'

model_state = torch.load(PYTORCH_MODEL, weights_only=False, map_location='cpu')
model_state['config']['order'] = 1
model_state['config']['accum'] = 'sum'
manager = Manager(
    model_state['config'],
    'cpu',
    model_state['src_lang'],
    model_state['tgt_lang'],
    PYTORCH_MODEL,
    'data/en-de.vocab',
    'data/en-de.model',
)
manager.model.load_state_dict(model_state['state_dict'])

knn_model = KNNModel(manager, n_neighbors=10)
if os.path.exists(KNN_MODEL):
    knn_model.load(KNN_MODEL)
else:
    knn_model.fit()
    knn_model.save(KNN_MODEL)

routes = web.RouteTableDef()


@routes.post('/neighbors')
async def knn_handler(request: web.Request) -> web.Response:
    args = await request.json()
    neighbors = knn_model.kneighbors(args.get('string'))
    return web.json_response({'neighbors': neighbors})


@routes.post('/translate')
async def translate_handler(request: web.Request) -> web.Response:
    args = await request.json()
    output, scores = translate(args.get('string'), manager, conf_type='grad')
    return web.json_response({'scores': scores, 'output': output})


async def init_app(enable_cors: bool = False) -> web.Application:
    app = web.Application()
    app.add_routes(routes)
    if enable_cors:
        cors = aiohttp_cors.setup(
            app,
            defaults={
                '*': aiohttp_cors.ResourceOptions(
                    allow_credentials=True,
                    expose_headers='*',
                    allow_headers='*',
                )
            },
        )
        for route in app.router.routes():
            cors.add(route)
    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default='0.0.0.0')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--dev-env', action='store_true')
    args = parser.parse_args()

    web.run_app(init_app(args.dev_env), host=args.host, port=args.port)

import argparse
import logging
import os

import aiohttp_cors
import torch
from aiohttp import web

from translation.knn import KNNModel
from translation.manager import Manager
from translation.translate import translate

PYTORCH_MODEL = 'data/en-de.pt'
KNN_MODEL = 'data/neighbors.pickle'

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    level=logging.INFO,
    handlers=[logging.FileHandler('data/api_server.log'), logging.StreamHandler()],
)

logging.info('Loading PyTorch Model...')
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
logging.info('KNN Model Loaded: ' + PYTORCH_MODEL)

knn_model = KNNModel(manager, 'data/en-de.freq', n_neighbors=5)
if os.path.exists(KNN_MODEL):
    logging.info('Loading KNN Model...')
    knn_model.load(KNN_MODEL)
    logging.info('PyTorch Model Loaded: ' + KNN_MODEL)
else:
    logging.info('Training KNN Model...')
    knn_model.fit()
    knn_model.save(KNN_MODEL)
    logging.info('KNN Model Saved: ' + KNN_MODEL)

routes = web.RouteTableDef()


@routes.post('/neighbors')
async def neighbors_handler(request: web.Request) -> web.Response:
    args = await request.json()
    neighbors = knn_model.kneighbors(args.get('string'))
    logging.info(f'POST /neighbors "{args.get('string')}"')
    logging.info(f'  Neighbors: {neighbors}')
    return web.json_response({'neighbors': neighbors})


@routes.post('/translate')
async def translate_handler(request: web.Request) -> web.Response:
    args = await request.json()
    output, scores = translate(args.get('string'), manager, conf_type='grad')
    counts = {word: int(knn_model.freq.get(word, 0)) for word, _ in scores[1:-1]}
    logging.info(f'POST /translate "{args.get('string')}"')
    logging.info(f'  Output: {output}')
    logging.info(f'  Scores: {scores}')
    logging.info(f'  Counts: {counts}')
    return web.json_response({'scores': scores, 'counts': counts, 'output': output})


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

    logging.info('Initializing API Server...')
    web.run_app(init_app(args.dev_env), host=args.host, port=args.port)

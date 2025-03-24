import argparse
import logging
import os

import aiohttp_cors
import torch
from aiohttp import web

from translation.knn import KNNModel
from translation.manager import Manager
from translation.translate import translate

NMT_MODEL = 'data/en-de.pt'
KNN_MODEL = 'data/faiss_index.ivf'

logging.basicConfig(
    format='[%(asctime)s %(levelname)s] %(message)s',
    level=logging.INFO,
    handlers=[logging.FileHandler('data/api_server.log'), logging.StreamHandler()],
)

logging.info('Loading NMT Model: ' + NMT_MODEL)
model_state = torch.load(NMT_MODEL, weights_only=False, map_location='cpu')
model_state['config']['order'] = 1
model_state['config']['accum'] = 'sum'
manager = Manager(
    model_state['config'],
    'cpu',
    model_state['src_lang'],
    model_state['tgt_lang'],
    NMT_MODEL,
    'data/en-de.vocab',
    'data/en-de.model',
)
manager.model.load_state_dict(model_state['state_dict'])

knn_model = KNNModel(manager, 'data/en-de.freq')
if os.path.exists(KNN_MODEL):
    logging.info('Loading KNN Model: ' + KNN_MODEL)
    knn_model.load(KNN_MODEL)
else:
    logging.info('Fitting KNN Model: ' + KNN_MODEL)
    knn_model.build_index()
    knn_model.save(KNN_MODEL)

routes = web.RouteTableDef()


@routes.get('/ntotal')
async def ntotal_handler(request: web.Request) -> web.Response:
    return web.json_response({'ntotal': knn_model.faiss_index.ntotal})


@routes.post('/neighbors')
async def neighbors_handler(request: web.Request) -> web.Response:
    args = await request.json()
    input_string = args.get('string')
    collect_data = args.get('send_data')
    if input_string is None:
        return web.json_response({'error': 'missing "string" parameter'}, status=400)
    neighbors = knn_model.search(
        input_string,
        n_neighbors=args.get('n_neighbors', 5),
        restrict_vocab=args.get('restrict_vocab'),
    )
    if collect_data is None or collect_data:
        logging.info(f'\x1b[33;20mPOST /neighbors\x1b[0m "{input_string}"')
        logging.info(f'Neighbors: {neighbors}')
    return web.json_response({'neighbors': neighbors})


@routes.post('/translate')
async def translate_handler(request: web.Request) -> web.Response:
    args = await request.json()
    input_string = args.get('string')
    collect_data = args.get('send_data')
    if input_string is None:
        return web.json_response({'error': 'missing "string" parameter'}, status=400)
    output, scores = translate(input_string, manager, conf_type='grad')
    counts = {word: knn_model.word_to_freq.get(word, 0) for word, _ in scores[1:-1]}
    if collect_data is None or collect_data:
        logging.info(f'\x1b[33;20mPOST /translate\x1b[0m "{input_string}"')
        logging.info(f'Output: {output}')
        logging.info(f'Scores: {[(word, f'{score:.2f}') for word, score in scores]}')
    return web.json_response({'output': output, 'scores': scores, 'counts': counts})


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

    logging.info('Starting API Server')
    web.run_app(init_app(args.dev_env), host=args.host, port=args.port)

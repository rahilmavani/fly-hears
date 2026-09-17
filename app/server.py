"""Serve the local microphone demo and recorded activity replay."""

import argparse
import io
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import librosa
import numpy as np
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from app.inference import Predictor
from app.audio_checks import NoSpeechError, require_speech
from scripts import artifacts, encode

APP_DIR = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024


def sample_recordings():
    directory = artifacts.DATA / 'fsdd' / 'recordings'
    if not directory.is_dir():
        directory = artifacts.ROOT / 'artifacts' / 'demo' / 'recordings'
    recordings = sorted(directory.glob('*.wav'))
    rng = np.random.default_rng(3)
    samples = []
    for digit in range(10):
        candidates = [path for path in recordings if path.name.startswith(f'{digit}_')]
        if candidates:
            samples.append(candidates[int(rng.integers(len(candidates)))])
    return samples


def decode_audio(raw):
    audio, sample_rate = librosa.load(io.BytesIO(raw), sr=encode.SR)
    if len(audio) < encode.SR // 10:
        raise ValueError('The recording is too short. Say one complete digit.')
    if len(audio) > encode.SR * 3:
        raise ValueError('Use a recording of one digit, no longer than three seconds.')
    if not np.isfinite(audio).all():
        raise ValueError('The recording contains invalid audio values.')
    require_speech(audio, sample_rate)
    return audio, sample_rate


def create_app(run_dir):
    @asynccontextmanager
    async def lifespan(application):
        application.state.samples = sample_recordings()
        try:
            application.state.predictor = Predictor(run_dir)
            application.state.problem = None
        except (ValueError, OSError, KeyError) as error:
            application.state.predictor = None
            application.state.problem = str(error)
            print(f'Prediction unavailable: {error}', flush=True)
        yield

    application = FastAPI(title='The Fly Hears', lifespan=lifespan)
    application.state.samples = []
    application.state.predictor = None
    application.state.problem = 'The model has not been loaded.'
    application.mount('/static', StaticFiles(directory=APP_DIR / 'static'), name='static')

    @application.get('/')
    def index():
        return HTMLResponse((APP_DIR / 'index.html').read_text(), headers={'Cache-Control': 'no-store'})

    @application.get('/status')
    def status():
        return JSONResponse({'ready': application.state.predictor is not None,
                             'error': application.state.problem}, headers={'Cache-Control': 'no-store'})

    @application.get('/results')
    def results():
        predictor = application.state.predictor
        return predictor.results if predictor is not None else {}

    @application.get('/samples')
    def samples():
        entries = [
            {'name': path.name, 'digit': int(path.name[0]), 'speaker': path.name.split('_')[1],
             'url': f"/recording/{quote(path.name, safe='')}"}
            for path in application.state.samples
        ]
        return JSONResponse(entries, headers={'Cache-Control': 'no-store'})

    @application.get('/recording/{name}')
    def recording(name: str):
        path = next((path for path in application.state.samples if path.name == name), None)
        if path is None:
            raise HTTPException(404, 'Recording unavailable. Refresh the sample list.')
        return FileResponse(path, media_type='audio/wav', headers={'Cache-Control': 'no-store'})

    @application.get('/sample/{index}')
    def obsolete_sample(index: int):
        return JSONResponse({'error': 'The sample links have changed. Refresh this page.'},
                            status_code=410, headers={'Cache-Control': 'no-store'})

    @application.post('/predict')
    async def predict(file: UploadFile = File(...)):
        predictor = application.state.predictor
        if predictor is None:
            return JSONResponse({'error': application.state.problem}, status_code=503)
        raw = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(raw) > MAX_UPLOAD_BYTES:
            return JSONResponse({'error': 'The audio file is too large.'}, status_code=413)
        try:
            audio, sample_rate = await run_in_threadpool(decode_audio, raw)
        except NoSpeechError as error:
            return JSONResponse({'error': str(error), 'code': 'no_speech'}, status_code=400)
        except ValueError as error:
            return JSONResponse({'error': str(error)}, status_code=400)
        except Exception:
            return JSONResponse({'error': 'Could not read the recording. Use a WAV file of one digit, 0.1–3 seconds long.'}, status_code=400)
        try:
            result = await run_in_threadpool(predictor.predict, audio, sample_rate)
            return JSONResponse(result)
        except ValueError as error:
            return JSONResponse({'error': str(error)}, status_code=400)

    return application


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8001)
    args = parser.parse_args()
    uvicorn.run(create_app(args.run), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()

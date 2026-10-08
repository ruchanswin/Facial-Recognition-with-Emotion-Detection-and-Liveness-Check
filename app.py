import gradio as gr
from fastapi import FastAPI

from gradio_app import demo


app = gr.mount_gradio_app(FastAPI(), demo, path="/")

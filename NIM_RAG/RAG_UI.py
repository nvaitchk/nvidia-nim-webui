import time
import gradio as gr
from inference import rag_func

# Force Dark Mode
js_func_darkmode = """
function refresh() {
    // Force Dark Mode
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""
# Define a class for functions
class func:
    def __init__(self):
        self.images_list=[]
        self.rag_fun = rag_func()
        
    def generate_response(self, question, backend, address, port, max_tokens, temperature, top_p, frequency_penalty, presence_penalty):
        max_tokens = max_tokens
        temperature = temperature
        top_p = top_p
        response = self.rag_fun.infer_request(question, backend, address, port, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
        return response

    def upload_doc(self, url):
        self.rag_fun.doc_process(url)
        return "Uploaded Successfully"

    
# css
css = """
.disclaimer.label.textarea {input-border-width: 0px !important}
footer {display: none !important}
"""

# Define Theme (Color, Font etc.)
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(c100="#e3f1cc", c200="#c8e399", c300="#acd566", c400="#91c732", c50="#f1f8e5", c500="#83c019", c600="#76b900", c700="#6aa600", c800="#5e9400", c900="#528100", c950="#3b5c00"),
    secondary_hue=gr.themes.Color(c100="#f0f0f0", c200="#b4b4b4", c300="#787878", c400="#646464", c50="#ffffff", c500="#3c3c3c", c600="#282828", c700="#202020", c800="#181818", c900="#101010", c950="#0c0c0c"),
    neutral_hue=gr.themes.Color(c100="#f0f0f0", c200="#b4b4b4", c300="#787878", c400="#646464", c50="#ffffff", c500="#3c3c3c", c600="#282828", c700="#202020", c800="#181818", c900="#101010", c950="#0c0c0c"),
    font=[gr.themes.GoogleFont('JetBrains Mono'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=[gr.themes.GoogleFont('JetBrains Mono'), 'ui-monospace', 'Consolas', 'monospace'],
).set(
    block_title_text_color_dark='*secondary_50',
    button_primary_text_color_dark='*secondary_950',
    block_label_background_fill_dark='*block_background_fill',
    checkbox_label_background_fill_selected_dark='*checkbox_label_background_fill',
    input_border_color_dark='*secondary_200',
    input_border_width_dark='0.5px'
)

# Gradio Main Body
with gr.Blocks(title="NVIDIA RAG NIM UI", theme=theme, css=css, js=js_func_darkmode) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(value="./img/nv_logo.png", label=None, show_label=False, interactive=False, show_download_button=False, elem_classes="logo")
        with gr.Column(scale=8):
            gr.Markdown("""
            # Llama-3-8B-Instruct
            AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or indecent. By testing this model, you assume the risk of any harm caused by any response or output of the model. Please do not upload any confidential information or personal data. Your use is logged for security.
            """)
  
    with gr.Row():
        with gr.Column(scale=5, variant="panel"):
            gr.Markdown("""
            # Input
            """)
            with gr.Row():
                backend = gr.Radio(["NIM", "NVIDIA AI Endpoint"], value="NVIDIA AI Endpoint", label="LLM Service Backend", scale=6)
            with gr.Row(visible=False) as address_port:
                address = gr.Textbox(value="0.0.0.0", label="Service Address", placeholder="0.0.0.0", visible=True)
                port = gr.Textbox(value="8000", label="Port", placeholder="8000", visible=True)
            
            question = gr.Textbox(value="What is NIM?", label="Prompt", placeholder="Write a limmerick about the wonders of GPU computing.", lines=3)
            
            with gr.Row():
                max_tokens = gr.Slider(value=1024, minimum=1, maximum=2048, label="Max Tokens", step=1)
            with gr.Row():
                temperature = gr.Slider(value=0.2, minimum=0.1, maximum=1, label="Temperature", step=0.01)
                top_p = gr.Slider(value=0.7, minimum=0.1, maximum=1, label="Top P", step=0.01)
            with gr.Row(visible=False) as nim_extra_params:
                frequency_penalty = gr.Slider(value=0, minimum=-2, maximum=2, label="Frequency Penalty", step=0.1, visible=True)
                presence_penalty = gr.Slider(value=0, minimum=-2, maximum=2, label="Presence Penalty", step=0.1, visible=True)
                
            def update_vis(backend):
                if backend=="NIM":
                    return [gr.Row(visible=bool(1)), gr.Row(visible=bool(1))]
                else:
                    return [gr.Row(visible=bool(0)), gr.Row(visible=bool(0))]
            backend.change(update_vis, backend, [address_port, nim_extra_params])
            
            btn_run = gr.Button("Run", variant="primary")
            
            url = gr.Textbox(label="Document URL", placeholder="https://developer.nvidia.com/blog/nvidia-nim-offers-optimized-inference-microservices-for-deploying-ai-models-at-scale/", lines=1)
            notice = gr.Textbox(label="Notice", placeholder="", lines=1, interactive=False)
            btn_upload = gr.Button("Upload", variant="primary")
        with gr.Column(scale=5, variant="panel"):
            gr.Markdown("""
            # Output
            """)
            chatbox = gr.Textbox(label="Response", placeholder="", lines=3)

    # Run Button
    fun = func()
    btn_run.click(fn=fun.generate_response, queue=True, inputs=[question, backend, address, port, max_tokens, temperature, top_p, frequency_penalty, presence_penalty], outputs=[chatbox])
    btn_upload.click(fn=fun.upload_doc, inputs=[url], outputs=[notice])
    Live=True


# Launch Gradio App
demo.launch(share=False, debug=False, favicon_path='./img/favicon.ico', server_name="0.0.0.0")

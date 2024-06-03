import gradio as gr
from inference import infer_request

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
        
    def generate_image(self, positive_prompt, negative_prompt, sampler, seed, inference_step, guidance_scale, address, port):
        img_np, seed_used, url = infer_request(positive_prompt, negative_prompt, sampler, seed, inference_step, guidance_scale, address, port)
        received = [img_np, str(seed_used)]
        self.images_list.append(received)
        images = self.images_list
        return images
    
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
with gr.Blocks(title="NVIDIA SDXL NIM UI", theme=theme, css=css, js=js_func_darkmode) as demo:
    with gr.Row():
        with gr.Column(scale=2):
            image = gr.Image(value="./img/nv_logo.png", label=None, show_label=False, interactive=False, show_download_button=False, elem_classes="logo")
        with gr.Column(scale=8):
            gr.Markdown("""
            # Stable Diffusion XL
            AI models generate responses and outputs based on complex algorithms and machine learning techniques, and those responses or outputs may be inaccurate or indecent. By testing this model, you assume the risk of any harm caused by any response or output of the model. Please do not upload any confidential information or personal data. Your use is logged for security.
            """)
  
    with gr.Row():
        with gr.Column(scale=5, variant="panel"):
            gr.Markdown("""
            # Input
            """)
            positive_prompt = gr.Textbox(label="Positive prompt", placeholder="a photo of an astronaut riding a horse on mars", lines=3)
            negative_prompt = gr.Textbox(label="Negative prompt", placeholder="Optional.")
            with gr.Row():
                sampler = gr.Radio(["DDIM", "DPM", "LMS", "EulerA"], value="DPM", label="Sampler", scale=6)
                seed = gr.Number(label="Seed", scale=4, min_width=3)
            with gr.Row():    
                inference_step = gr.Slider(minimum=5, maximum=50, label="Inference Steps", step=1)
                guidance_scale = gr.Slider(minimum=1.5, maximum=9, label="Guidance Scale", step=0.1)
            with gr.Row():
                address = gr.Textbox(label="Service Address", placeholder="127.0.0.1")
                port = gr.Textbox(label="Port", placeholder="8003")
            btn = gr.Button("Run", variant="primary")
        with gr.Column(scale=5, variant="panel"):
            gr.Markdown("""
            # Output
            """)
            gallery = gr.Gallery(container=True, 
            #value=images_gallery, 
            label=None, show_label=False, preview=True, height="600px", columns=2, rows=2, interactive=False)
    # Run Button
    fun = func()
    btn.click(fn=fun.generate_image, inputs=[positive_prompt, negative_prompt, sampler, seed, inference_step, guidance_scale, address, port], outputs=[gallery])
    Live=True


# Launch Gradio App
demo.launch(share=True, debug=False, favicon_path='./img/favicon.ico', server_name="0.0.0.0")

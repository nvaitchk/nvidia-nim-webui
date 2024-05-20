import json
import base64
import requests
from PIL import Image
import numpy
import io



def infer_request(positive_prompt, negative_prompt, sampler, seed, inference_step, guidance_scale, address, port):
    print("Debug: " + str([positive_prompt, negative_prompt, sampler, seed, inference_step, guidance_scale, address, port]))
    sampler_dict = {
      "DDIM": "DDIM",
      "DPM": "K_DPM_2_ANCESTRAL",
      "LMS": "K_LMS", 
      "EulerA": "K_EULER_ANCESTRAL"
    }
    payload = json.dumps({
        "height": 1024,
        "width": 1024,
        "text_prompts": [
            {
                "weight": 1,
                "text": positive_prompt
            },
            {
                "weight": -1,
                "text": negative_prompt
            }
        ],
        "cfg_scale": guidance_scale,
        "clip_guidance_preset": "NONE",
        "sampler": sampler_dict[sampler],
        "samples": 1,
        "seed": seed,
        "steps": inference_step,
        "style_preset": "none"
    })
    
    
    url = "http://" + address + ":" + port + "/infer"
    response = requests.post(url,  data=payload)
    #response.raise_for_status()
    
    #Post-Process Response
    data = response.json()
    if data != None:
    	img_base64 = data['artifacts'][0]["base64"]
    	seed_used = data['artifacts'][0]["seed"]

    	img_decode = io.BytesIO(base64.b64decode(img_base64))
    
    	img = Image.open(img_decode)
    	img_np = numpy.array(img)

    	return img_np, seed_used, url
    else:
        return []

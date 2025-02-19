import requests
import base64
import json
from mimetypes import guess_type
import os
import argparse
import yaml
import pathlib

def convert_image_to_base64(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found
    # Read and encode the image file
    with open(image_path, 'rb') as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')
    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"

def send_multimodal_request(api_key, messages):
    # Prepare the headers and data
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": "gpt-4o",  # Use a model that supports vision
        "stream": False,
        "max_tokens": 1000,
        "messages": messages
    }
    
    try:
        # Send the request
        response = requests.post(
            "https://models-proxy.stepfun-inc.com/v1/chat/completions",  # Use the correct API endpoint
            headers=headers,
            data=json.dumps(data)
        )
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes
        return response.json()
    except requests.RequestException as e:
        print(f"Request error: {e}")
        return None

# Example usage
api_key = "ak-39d8efgh45i6jkl23mno78pqrs12tuv4k5"

def query_gpt(file_name, image_path, questions):
    image_url = convert_image_to_base64(image_path)
    
    # Initialize the conversation with the image and the first question
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
                },
                {
                    "type": "text",
                    "text": questions
                }
            ]
        }
    ]
    
    # List to store answers for this image
    image_answers = []
    
    # Process each question
    for j, question in enumerate([questions]):
        if j > 0:
            # Add the previous answer and the new question to the messages
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    }
                ]
            })
        
        response = send_multimodal_request(api_key, messages)
        if response is not None:
            
            # Extract the assistant's answer and add it to the answers list
            answer = response['choices'][0]['message']['content']
            image_answers.append({
                "question": question,
                "answer": answer
            })
            print(f"Response to question {j+1} for {file_name}:", answer)
            
            # Add the assistant's response to the messages for the next question
            messages.append({
                "role": "assistant",
                "content": answer
            })
    
    # Add the answers for this image to the overall list
    return image_answers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to config file', type=str, default='./example_config.yml')
    args = parser.parse_args()
    if args.config is not None:
        with open(args.config, 'r') as f:
            try:
                cfg = yaml.safe_load(f)
            except yaml.YAMLError as e:
                print(e)
    
    for key in vars(args):
        cfg[key] = vars(args)[key]
    output_path = pathlib.Path(cfg['output_path'])

    target_prompt = cfg['text_prompt']
    questions = f"Knowing that {target_prompt} has a similar appearance to the input image, describe the details of {target_prompt}'s appearance using content in the following format, e.g., “The nine-grid image shows different views of a whale. The whale is characterized by a huge, elongated body and a distinct head. The whale's head is characterized by a large, curved mouth and a pair of eyes. The body is streamlined and tapers toward a broad, flat tail. The mesh exhibits a faceted, low polygon style that gives the whale a geometric and simplified appearance. The absence of texture and color in the mesh indicates that it is a basic 3D model that can be used as a basis for further design and development. "

    image_path = os.path.join(output_path, "source_mesh.png")
    answer = query_gpt(cfg['mesh'], image_path, questions)

    final_answer = answer[-1]['answer'] + f"[PATCH 1] The front view of the 3D {target_prompt} mesh, [PATCH 2] The back view of the 3D {target_prompt} mesh, [PATCH 3] The right view of the 3D {target_prompt} mesh, [PATCH 4] The left view of the 3D {target_prompt} mesh, [PATCH 5] The front right view of the 3D {target_prompt} mesh, [PATCH 6] The front left view of the 3D {target_prompt} mesh, [PATCH 7] The back righ view of the 3D {target_prompt} mesh, [PATCH 8] The back left view of the 3D {target_prompt} mesh, [PATCH 9] The top view of the 3D {target_prompt} mesh."
    cfg['target_prompt'] = final_answer

    # 保存修改后的配置到 YAML 文件
    with open(args.config, 'w') as f:
        yaml.dump(cfg, f)

    print("Done!")